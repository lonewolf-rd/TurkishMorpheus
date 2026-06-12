import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.sigmorphon import build_segmenters
from src.benchmarker.benchmarks.trmmlu_eval import find_morpheus_checkpoint


MORPHSCORE_REPO_ID = "catherinearnett/morphscore"
MORPHSCORE_FILENAME = "turkish_data.csv"


def load_morphscore_turkish(
        local_path: Path,
        max_word_len: int = 30,
) -> pd.DataFrame:
    if not local_path.exists():
        from huggingface_hub import hf_hub_download
        global_logger.info(
            f"[morphscore_eval](load_morphscore_turkish) Downloading "
            f"{MORPHSCORE_FILENAME} from {MORPHSCORE_REPO_ID}"
        )
        downloaded = hf_hub_download(
            repo_id=MORPHSCORE_REPO_ID,
            filename=MORPHSCORE_FILENAME,
            repo_type="dataset",
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded, local_path)

    df = pd.read_csv(local_path)
    n_raw = len(df)

    df = df.dropna(subset=["wordform", "stem"])
    df["wordform"] = df["wordform"].astype(str)
    df["stem"] = df["stem"].astype(str)

    if "unique" in df.columns and (df["unique"] == "unique").any():
        df = df[df["unique"] == "unique"]

    if "lemma" in df.columns:
        df = df[df["stem"] == df["lemma"].astype(str)]

    df = df[~df["wordform"].str.contains(r"\d", regex=True)]
    df = df[~df["wordform"].str.contains(r"\s", regex=True)]
    df = df[df["wordform"].str.len().between(2, max_word_len)]
    df = df.drop_duplicates(subset=["wordform"]).reset_index(drop=True)

    global_logger.info(
        f"[morphscore_eval](load_morphscore_turkish) {n_raw:,} raw rows -> "
        f"{len(df):,} filtered unique wordforms"
    )
    return df


def gold_morphemes(row: pd.Series) -> List[str]:
    parts: List[str] = []
    for col in ("preceding_part", "stem", "following_part"):
        v = row.get(col)
        if isinstance(v, str) and v and v.lower() != "nan":
            parts.append(v)
    return parts


def boundaries_from_parts(parts: List[str]) -> Set[int]:
    bnds: Set[int] = set()
    pos = 0
    for p in parts[:-1]:
        pos += len(p)
        bnds.add(pos)
    return bnds


def evaluate(
        df: pd.DataFrame,
        segmenters: Dict[str, Any],
        output_dir: Path,
        exclude_single_tok: bool = False,
        single_tok_point: float = 0.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "recall_sum": 0.0,
        "precision_sum": 0.0,
        "weighted_recall_sum": 0.0,
        "weighted_precision_sum": 0.0,
        "weight_sum": 0.0,
        "micro_matched": 0,
        "micro_gold": 0,
        "micro_pred": 0,
        "n_words": 0,
        "n_single_token": 0,
        "n_len_mismatch": 0,
        "n_failed": 0,
    })
    detail_rows: List[Dict] = []

    n_skipped_mismatch = 0
    n_skipped_single_morpheme = 0

    for _, row in df.iterrows():
        word = row["wordform"]
        parts = gold_morphemes(row)

        if len(parts) < 2:
            n_skipped_single_morpheme += 1
            continue
        if "".join(parts) != word:
            n_skipped_mismatch += 1
            continue

        gold_bnds = boundaries_from_parts(parts)
        freq = float(row.get("word_freq", 1.0) or 1.0)

        detail = {
            "wordform": word,
            "gold": " | ".join(parts),
            "n_gold_boundaries": len(gold_bnds),
            "word_freq": freq,
        }

        for name, segmenter in segmenters.items():
            try:
                segs = segmenter(word)
            except Exception as e:
                stats[name]["n_failed"] += 1
                global_logger.warning(
                    f"[morphscore_eval](evaluate) {name} failed on '{word}': {e}"
                )
                continue

            s = stats[name]

            if len(segs) <= 1:
                s["n_single_token"] += 1
                if exclude_single_tok:
                    detail[f"{name}_seg"] = " | ".join(segs)
                    detail[f"{name}_recall"] = ""
                    continue
                recall = single_tok_point
                precision = single_tok_point
                matched = 0
                n_pred = 0
            else:
                pred_bnds = boundaries_from_parts(segs)
                matched = len(gold_bnds & pred_bnds)
                n_pred = len(pred_bnds)
                recall = matched / len(gold_bnds)
                precision = matched / max(n_pred, 1)

            s["recall_sum"] += recall
            s["precision_sum"] += precision
            s["weighted_recall_sum"] += recall * freq
            s["weighted_precision_sum"] += precision * freq
            s["weight_sum"] += freq
            s["micro_matched"] += matched
            s["micro_gold"] += len(gold_bnds)
            s["micro_pred"] += n_pred
            s["n_words"] += 1
            if sum(len(x) for x in segs) != len(word):
                s["n_len_mismatch"] += 1

            detail[f"{name}_seg"] = " | ".join(segs)
            detail[f"{name}_recall"] = round(recall, 4)

        detail_rows.append(detail)

    global_logger.info(
        f"[morphscore_eval](evaluate) Skipped: "
        f"{n_skipped_single_morpheme:,} single-morpheme, "
        f"{n_skipped_mismatch:,} part/wordform mismatch"
    )

    summary_rows: List[Dict] = []
    for name, s in stats.items():
        n = max(s["n_words"], 1)
        w = max(s["weight_sum"], 1e-9)
        mean_recall = s["recall_sum"] / n
        mean_precision = s["precision_sum"] / n
        macro_f1 = (
            2 * mean_precision * mean_recall / max(mean_precision + mean_recall, 1e-9)
        )
        micro_p = s["micro_matched"] / max(s["micro_pred"], 1)
        micro_r = s["micro_matched"] / max(s["micro_gold"], 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
        summary_rows.append({
            "model": name,
            "n_words": int(s["n_words"]),
            "morphscore_recall": round(mean_recall, 4),
            "morphscore_precision": round(mean_precision, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "freq_weighted_recall": round(s["weighted_recall_sum"] / w, 4),
            "freq_weighted_precision": round(s["weighted_precision_sum"] / w, 4),
            "single_token_rate": round(s["n_single_token"] / n, 4),
            "len_mismatch_rate": round(s["n_len_mismatch"] / n, 4),
            "n_failed": int(s["n_failed"]),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("morphscore_recall", ascending=False)
        .reset_index(drop=True)
    )
    summary_path = output_dir / "morphscore_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    global_logger.info(f"[morphscore_eval](evaluate) Summary: {summary_path}")

    details_df = pd.DataFrame(detail_rows)
    details_path = output_dir / "morphscore_details.csv"
    details_df.to_csv(details_path, index=False)
    global_logger.info(f"[morphscore_eval](evaluate) Per-word details: {details_path}")

    print()
    print("=" * 78)
    print("MORPHSCORE v2 — Turkish gold morpheme boundaries (UD_Turkish-Kenet)")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print("=" * 78)
    print(f"Artifacts: {output_dir.absolute()}")

    return summary_df, details_df


def run(
        classical_vocab: int = 64000,
        exclude_single_tok: bool = False,
        single_tok_point: float = 0.0,
) -> bool:
    base = Path(__file__).resolve().parents[3]
    artifacts = base / "src" / "model_development" / "artifacts"
    data_path = base / "data" / "morphscore" / MORPHSCORE_FILENAME
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "morphscore"

    checkpoint = find_morpheus_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[morphscore_eval](run) No Morpheus checkpoint found.")
        return False

    df = load_morphscore_turkish(data_path)
    if df.empty:
        global_logger.error("[morphscore_eval](run) No usable MorphScore rows.")
        return False

    segmenters = build_segmenters(
        checkpoint_path=str(checkpoint),
        morfessor_path=str(artifacts / "tokenizers" / "classical" / "morfessor_model.bin"),
        benchmarker_results_dir=str(artifacts / "tokenizers" / "classical"),
        preferred_classical_vocab=classical_vocab,
    )

    evaluate(
        df,
        segmenters,
        output_dir,
        exclude_single_tok=exclude_single_tok,
        single_tok_point=single_tok_point,
    )
    return True


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.benchmarks.morphscore_eval")
    parser.add_argument("--classical-vocab", type=int, default=64000)
    parser.add_argument("--exclude-single-tok", action="store_true")
    parser.add_argument("--single-tok-point", type=float, default=0.0)
    args = parser.parse_args()

    ok = run(
        classical_vocab=args.classical_vocab,
        exclude_single_tok=args.exclude_single_tok,
        single_tok_point=args.single_tok_point,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
