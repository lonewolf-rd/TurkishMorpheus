import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.paper import discover_classical_tokenizers
from src.benchmarker.benchmarks.trmmlu_eval import find_morpheus_checkpoint
from src.benchmarker.benchmarks.morphscore_eval import load_morphscore_turkish


RoundtripFn = Callable[[str], str]


def make_classical_roundtrip(wrapper) -> RoundtripFn:
    model = wrapper.model
    if wrapper.kind in ("bpe", "byte_bpe", "unigram", "wordpiece"):
        return lambda w: model.decode(model.encode(w, out_type=int))
    return lambda w: model.decode(model.encode(w).ids)


def make_morpheus_roundtrip(checkpoint_path: str, tokenizer_dir: str) -> RoundtripFn:
    from src.model_development.model.morpheus import Morpheus
    from src.model_development.training.trainer import TrainingConfig
    from src.model_development.tokenization.morpheus_tokenizer import MorpheusTokenizer

    sys.modules["__main__"].TrainingConfig = TrainingConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tok = MorpheusTokenizer.load(tokenizer_dir, morpheus_model=model, device=device)
    return lambda w: tok.decode(tok.encode(w, add_special_tokens=False))


def build_roundtrip_fns(
        artifacts_dir: Path,
        preferred_classical_vocab: int,
) -> Dict[str, RoundtripFn]:
    fns: Dict[str, RoundtripFn] = {}

    classical_dir = artifacts_dir / "tokenizers" / "classical"
    for wrapper in discover_classical_tokenizers(str(classical_dir), preferred_classical_vocab):
        try:
            fns[wrapper.name.lower()] = make_classical_roundtrip(wrapper)
        except Exception as e:
            global_logger.error(f"[roundtrip_eval] {wrapper.name} roundtrip setup failed: {e}")

    ckpt = find_morpheus_checkpoint(artifacts_dir / "checkpoints")
    morpheus_dir = artifacts_dir / "tokenizers" / "morpheus_50k"
    if ckpt and morpheus_dir.exists():
        try:
            fns["morpheus"] = make_morpheus_roundtrip(str(ckpt), str(morpheus_dir))
        except Exception as e:
            global_logger.error(f"[roundtrip_eval] Morpheus roundtrip setup failed: {e}")

    from src.benchmarker.metrics.external_tokenizers import load_turkish_tokenizer_or_none
    tt = load_turkish_tokenizer_or_none()
    if tt is not None:
        fns["turkish-tokenizer"] = lambda w: tt.tt.decode(tt.tt.encode(w))

    return fns


def evaluate(
        words: List[str],
        roundtrip_fns: Dict[str, RoundtripFn],
        output_dir: Path,
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for name, rt in roundtrip_fns.items():
        n_ok = 0
        n_total = 0
        failures: List[Tuple[str, str]] = []
        t0 = time.perf_counter()
        for w in words:
            try:
                out = rt(w).strip()
            except Exception:
                out = ""
            n_total += 1
            if out == w:
                n_ok += 1
            elif len(failures) < 200:
                failures.append((w, out))
        elapsed = time.perf_counter() - t0

        acc = n_ok / max(n_total, 1)
        rows.append({
            "tokenizer": name,
            "n_words": n_total,
            "roundtrip_acc": round(acc, 4),
            "roundtrip_fail_pct": round((1 - acc) * 100, 2),
            "n_fail": n_total - n_ok,
            "recon_words_per_sec": round(n_total / elapsed, 1) if elapsed > 0 else 0.0,
        })

        fail_path = output_dir / f"roundtrip_fail_{name}.csv"
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wordform", "reconstructed"])
            writer.writerows(failures)

        global_logger.info(
            f"[roundtrip_eval] {name}: acc={acc:.4f} fail={n_total - n_ok}/{n_total}"
        )

    rows.sort(key=lambda r: -r["roundtrip_acc"])
    summary_path = output_dir / "roundtrip_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("ROUNDTRIP RECONSTRUCTION  (decode(encode(w)) == w, inflected wordforms)")
    print("=" * 70)
    print(f"  {'tokenizer':<20s} {'n_words':>8s} {'acc':>8s} {'fail%':>7s} {'n_fail':>7s}")
    for r in rows:
        print(
            f"  {r['tokenizer']:<20s} {r['n_words']:>8,} {r['roundtrip_acc']:>8.4f} "
            f"{r['roundtrip_fail_pct']:>7.2f} {r['n_fail']:>7,}"
        )
    print("=" * 70)
    print(f"Summary CSV: {summary_path}")
    return rows


def run(classical_vocab: int = 64000) -> List[Dict]:
    base = Path(__file__).resolve().parents[3]
    artifacts = base / "src" / "model_development" / "artifacts"
    data_path = base / "data" / "morphscore" / "turkish_data.csv"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "roundtrip"

    df = load_morphscore_turkish(data_path)
    words = df["wordform"].astype(str).tolist()
    global_logger.info(f"[roundtrip_eval] {len(words):,} inflected wordforms")

    fns = build_roundtrip_fns(artifacts, classical_vocab)
    if not fns:
        global_logger.error("[roundtrip_eval] No tokenizers available.")
        return []

    return evaluate(words, fns, output_dir)


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.benchmarks.roundtrip_eval")
    parser.add_argument("--classical-vocab", type=int, default=64000)
    args = parser.parse_args()
    rows = run(classical_vocab=args.classical_vocab)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
