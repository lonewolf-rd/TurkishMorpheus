import sys
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.training.dataset import MorfessorWrapper
from src.model_development.training.trainer import TrainingConfig
from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.paper import (
    ClassicalTokenizerWrapper,
    discover_classical_tokenizers,
)

sys.modules["__main__"].TrainingConfig = TrainingConfig


VERB_SUFFIX_STRIPS = ("mak", "mek")

STRUCTURAL_FEATURES = {
    "V", "N", "ADJ", "ADV", "PRO",
    "DECL", "IND", "INTR", "POS", "NEG",
    "LGSPEC01", "LGSPEC02", "LGSPEC03", "LGSPEC04",
}


def lemma_to_root(lemma: str, is_verb: bool) -> str:
    if not is_verb:
        return lemma
    for suf in VERB_SUFFIX_STRIPS:
        if lemma.endswith(suf) and len(lemma) > len(suf) + 1:
            return lemma[: -len(suf)]
    return lemma


def count_morphological_features(features: List[str]) -> int:
    return sum(1 for f in features if f not in STRUCTURAL_FEATURES)


def load_sigmorphon_inflection_gold(
        gold_path: str,
        single_word_only: bool = True,
        min_inflected_len: int = 3,
        max_inflected_len: int = 30,
) -> List[Dict]:
    entries: List[Dict] = []
    skipped_multiword = 0
    skipped_length = 0

    path = Path(gold_path)
    if not path.exists():
        global_logger.error(f"[SIGMORPHON] Gold file not found: {gold_path}")
        return entries

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            lemma = parts[0].strip()
            inflected = parts[1].strip()
            features = [t.strip() for t in parts[2].split(";") if t.strip()]
            if not features:
                continue

            if single_word_only and (" " in inflected or " " in lemma):
                skipped_multiword += 1
                continue

            if not (min_inflected_len <= len(inflected) <= max_inflected_len):
                skipped_length += 1
                continue

            is_verb = features[0] == "V"
            is_noun = features[0] == "N"
            gold_root = lemma_to_root(lemma, is_verb)
            n_features = count_morphological_features(features)

            entries.append({
                "lemma": lemma,
                "lemma_root": gold_root,
                "inflected": inflected,
                "features": features,
                "n_morphological_features": n_features,
                "pos": features[0],
                "is_verb": is_verb,
                "is_noun": is_noun,
            })

    global_logger.info(
        f"[SIGMORPHON] Loaded {len(entries)} usable entries from {path} "
        f"(skipped {skipped_multiword} multi-word, {skipped_length} length-filtered)"
    )
    return entries


def lemma_preserved_as_prefix(segments: List[str], gold_root: str) -> bool:
    if not segments or not gold_root:
        return False
    gold_lower = turkish_lower(gold_root)
    accumulated = ""
    for seg in segments:
        accumulated += turkish_lower(seg)
        if accumulated == gold_lower:
            return True
        if len(accumulated) > len(gold_lower):
            return False
    return False


def root_in_segments(segments: List[str], gold_root: str) -> bool:
    gold_lower = turkish_lower(gold_root)
    return any(turkish_lower(s) == gold_lower for s in segments)


def first_segment_match(segments: List[str], gold_root: str) -> bool:
    if not segments:
        return False
    return turkish_lower(segments[0]) == turkish_lower(gold_root)


def morpheus_segment(
        model: Morpheus,
        helper: CharEncoderHelper,
        word: str,
        device: torch.device,
        max_word_len: int = 32,
        threshold: float = 0.5,
) -> List[str]:
    ids, flags, rl = helper.word_to_char_ids(word, max_len=max_word_len)
    char_ids = torch.tensor([ids], device=device)
    case_flags = torch.tensor([flags], device=device)
    real_lengths = torch.tensor([rl], device=device)

    with torch.no_grad():
        out = model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_lengths,
        )
    boundary_probs = out["boundary_probs"][0].cpu().tolist()

    n_chars = len(word)
    segments: List[str] = []
    current = word[0] if word else ""
    for i in range(1, n_chars):
        if i < len(boundary_probs) and boundary_probs[i] > threshold:
            segments.append(current)
            current = word[i]
        else:
            current += word[i]
    if current:
        segments.append(current)
    return segments


def evaluate(
        entries: List[Dict],
        segmenters: Dict[str, Any],
        output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_word_rows: List[Dict] = []
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "lemma_prefix_hits": 0,
        "root_in_segments_hits": 0,
        "first_segment_hits": 0,
        "expected_minus_actual_sum": 0,
        "abs_expected_minus_actual_sum": 0,
        "exact_count_hits": 0,
        "n_words": 0,
        "n_verb_words": 0,
        "n_noun_words": 0,
        "verb_lemma_prefix_hits": 0,
        "noun_lemma_prefix_hits": 0,
    })

    for entry in entries:
        word = entry["inflected"]
        gold_root = entry["lemma_root"]
        n_expected_suffixes = entry["n_morphological_features"]

        row = {
            "lemma": entry["lemma"],
            "lemma_root": gold_root,
            "inflected": word,
            "pos": entry["pos"],
            "features": ";".join(entry["features"]),
            "n_expected_suffixes": n_expected_suffixes,
        }

        for model_name, segmenter in segmenters.items():
            try:
                segs = segmenter(word)
            except Exception as e:
                global_logger.warning(f"[SIGMORPHON] {model_name} failed on '{word}': {e}")
                continue

            prefix_ok = lemma_preserved_as_prefix(segs, gold_root)
            root_ok = root_in_segments(segs, gold_root)
            first_ok = first_segment_match(segs, gold_root)
            n_segments = len(segs)
            n_actual_suffixes = max(0, n_segments - 1)
            count_diff = n_actual_suffixes - n_expected_suffixes

            row[f"{model_name}_segmentation"] = " | ".join(segs)
            row[f"{model_name}_n_segments"] = n_segments
            row[f"{model_name}_lemma_prefix_ok"] = prefix_ok
            row[f"{model_name}_root_in_segs"] = root_ok
            row[f"{model_name}_first_seg_ok"] = first_ok
            row[f"{model_name}_suffix_count_diff"] = count_diff

            s = stats[model_name]
            s["n_words"] += 1
            s["lemma_prefix_hits"] += int(prefix_ok)
            s["root_in_segments_hits"] += int(root_ok)
            s["first_segment_hits"] += int(first_ok)
            s["expected_minus_actual_sum"] += count_diff
            s["abs_expected_minus_actual_sum"] += abs(count_diff)
            s["exact_count_hits"] += int(count_diff == 0)
            if entry["is_verb"]:
                s["n_verb_words"] += 1
                s["verb_lemma_prefix_hits"] += int(prefix_ok)
            elif entry["is_noun"]:
                s["n_noun_words"] += 1
                s["noun_lemma_prefix_hits"] += int(prefix_ok)

        per_word_rows.append(row)

    details_df = pd.DataFrame(per_word_rows)
    details_path = out_dir / "sigmorphon_details.csv"
    details_df.to_csv(details_path, index=False)
    global_logger.info(f"[SIGMORPHON] Per-word details: {details_path}")

    summary_rows: List[Dict] = []
    for model_name, s in stats.items():
        n = max(s["n_words"], 1)
        n_v = max(s["n_verb_words"], 1)
        n_n = max(s["n_noun_words"], 1)
        summary_rows.append({
            "model": model_name,
            "n_words": s["n_words"],
            "lemma_prefix_rate": round(s["lemma_prefix_hits"] / n, 4),
            "verb_lemma_prefix_rate": round(s["verb_lemma_prefix_hits"] / n_v, 4),
            "noun_lemma_prefix_rate": round(s["noun_lemma_prefix_hits"] / n_n, 4),
            "root_in_segments_rate": round(s["root_in_segments_hits"] / n, 4),
            "first_segment_match_rate": round(s["first_segment_hits"] / n, 4),
            "exact_suffix_count_rate": round(s["exact_count_hits"] / n, 4),
            "mean_suffix_count_diff": round(s["expected_minus_actual_sum"] / n, 4),
            "mean_abs_suffix_count_diff": round(s["abs_expected_minus_actual_sum"] / n, 4),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("lemma_prefix_rate", ascending=False)
        .reset_index(drop=True)
    )
    summary_path = out_dir / "sigmorphon_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    global_logger.info(f"[SIGMORPHON] Summary: {summary_path}")

    long_rows: List[Dict] = []
    for _, row in summary_df.iterrows():
        for metric in (
            "lemma_prefix_rate", "verb_lemma_prefix_rate", "noun_lemma_prefix_rate",
            "root_in_segments_rate", "first_segment_match_rate",
            "exact_suffix_count_rate", "mean_suffix_count_diff", "mean_abs_suffix_count_diff",
        ):
            long_rows.append({
                "model": row["model"],
                "metric": metric,
                "value": row[metric],
                "n_words": row["n_words"],
            })
    long_df = pd.DataFrame(long_rows)
    long_path = out_dir / "sigmorphon_summary_long.csv"
    long_df.to_csv(long_path, index=False)
    global_logger.info(f"[SIGMORPHON] Long-format summary: {long_path}")

    print()
    print("=" * 78)
    print("SIGMORPHON 2022 Turkish Inflection — Lemma Preservation Benchmark")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print("=" * 78)
    print(f"\nArtifacts: {out_dir.absolute()}")

    return details_df, summary_df


def build_segmenters(
        checkpoint_path: str,
        morfessor_path: str,
        benchmarker_results_dir: Optional[str] = None,
        preferred_classical_vocab: int = 64000,
        device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = Morpheus(
        char_dim=cfg.char_dim,
        char_embed_dim=cfg.char_embed_dim,
        case_embed_dim=cfg.case_embed_dim,
        n_layers_encoder=cfg.n_layers_encoder,
        n_layers_detector=cfg.n_layers_detector,
        num_heads=cfg.num_heads,
        max_word_len=cfg.max_word_len,
        max_segs=cfg.max_segs,
        dropout=cfg.dropout,
        threshold=cfg.threshold,
        pos_weight=cfg.pos_weight,
        count_loss_w=getattr(cfg, "count_loss_w", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    helper = CharEncoderHelper()
    morfessor = MorfessorWrapper(morfessor_path)

    segmenters: Dict[str, Any] = {
        "morpheus": lambda w: morpheus_segment(model, helper, w, device, cfg.max_word_len),
        "morfessor": lambda w: morfessor.segment(w)[0],
    }

    if benchmarker_results_dir:
        classical = discover_classical_tokenizers(benchmarker_results_dir, preferred_classical_vocab)
        for tok in classical:
            segmenters[tok.name.lower()] = tok.segment

    global_logger.info(f"[SIGMORPHON] Loaded {len(segmenters)} segmenters: {list(segmenters.keys())}")
    return segmenters


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent

    gold_path = str(base / "data/sigmorphon_tr/tur.gold")
    checkpoint = str(base / "src/model_development/artifacts/checkpoints/turkish_morpheus_a100_best.pt")
    morfessor_path = str(base / "src/model_development/artifacts/tokenizers/classical/morfessor_model.bin")
    benchmarker_results = str(base / "src/model_development/artifacts/tokenizers/classical")
    output_dir = str(base / "src/benchmarker/results/paper_eval/sigmorphon")

    entries = load_sigmorphon_inflection_gold(gold_path)
    if not entries:
        print(f"[SIGMORPHON] No usable entries from {gold_path}")
        sys.exit(1)

    segmenters = build_segmenters(
        checkpoint_path=checkpoint,
        morfessor_path=morfessor_path,
        benchmarker_results_dir=benchmarker_results,
    )

    evaluate(entries, segmenters, output_dir)
