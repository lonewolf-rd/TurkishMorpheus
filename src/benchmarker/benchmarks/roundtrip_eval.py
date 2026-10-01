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


Codec = Tuple[Callable[[str], List[int]], Callable[[List[int]], str]]

STRESS_SENTENCES: List[str] = [
    "Evlerimizdekiler, kitapçısı muvaffakiyetsizleştiriciler.",
    "Ankara'da hava 3,5 °C; fiyat 12.000 ₺ (KDV dahil).",
    "“Yarın gelirim” dedi — ama gelmedi…",
    "ABD, AB ve TBMM'nin 2024'teki kararları %12 arttı.",
    "Detaylar için https://example.com/a?b=c&d=e adresine bakın.",
    "def topla(a, b):\n    return a + b\n",
    "if x:\n\ty = 1\n\n  z = 2",
    "Satır bir\r\nsatır iki\r\n",
    "  baştaki ve sondaki boşluklar  ",
    "iki  boşluk   üç boşluk",
    "Word WWW xyz QUIZ âlim îman kâr",
    "emoji 🙂 ve ± işareti, i̇ birleşik nokta",
    "no-break\u00a0space ve\u2009ince boşluk",
    "Çekoslovakyalılaştıramadıklarımızdanmışsınızcasına söyledi.",
]


def make_classical_codec(wrapper) -> Codec:
    model = wrapper.model
    if wrapper.kind in ("bpe", "byte_bpe", "unigram"):
        return (lambda w: model.encode(w, out_type=int),
                lambda ids: model.decode(ids))
    return (lambda w: model.encode(w).ids,
            lambda ids: model.decode(ids))


def make_morpheus_codec(checkpoint_path: str, tokenizer_dir: str) -> Codec:
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
    return (lambda w: tok.encode(w, add_special_tokens=False),
            lambda ids: tok.decode(ids))


def build_codecs(
        artifacts_dir: Path,
        preferred_classical_vocab: int,
) -> Dict[str, Codec]:
    codecs: Dict[str, Codec] = {}

    classical_dir = artifacts_dir / "tokenizers" / "classical"
    for wrapper in discover_classical_tokenizers(str(classical_dir), preferred_classical_vocab):
        try:
            codecs[wrapper.name.lower()] = make_classical_codec(wrapper)
        except Exception as e:
            global_logger.error(f"[roundtrip_eval] {wrapper.name} codec setup failed: {e}")

    ckpt = find_morpheus_checkpoint(artifacts_dir / "checkpoints")
    morpheus_dir = artifacts_dir / "tokenizers" / "morpheus_50k"
    if ckpt and morpheus_dir.exists():
        try:
            codecs["morpheus"] = make_morpheus_codec(str(ckpt), str(morpheus_dir))
        except Exception as e:
            global_logger.error(f"[roundtrip_eval] Morpheus codec setup failed: {e}")

    from src.benchmarker.metrics.external_tokenizers import load_turkish_tokenizer_or_none
    tt = load_turkish_tokenizer_or_none()
    if tt is not None:
        codecs["turkish-tokenizer"] = (lambda w: tt.tt.encode(w),
                                       lambda ids: tt.tt.decode(ids))

    return codecs


def evaluate(
        words: List[str],
        codecs: Dict[str, Codec],
        output_dir: Path,
        label: str = "words",
        strip: bool = True,
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    n_total = len(words)

    for name, (encode_fn, decode_fn) in codecs.items():
        ids_list: List = []
        for w in words:
            try:
                ids_list.append(encode_fn(w))
            except Exception:
                ids_list.append(None)

        t0 = time.perf_counter()
        outs: List[str] = []
        for ids in ids_list:
            if ids is None:
                outs.append("")
                continue
            try:
                out = decode_fn(ids)
                outs.append(out.strip() if strip else out)
            except Exception:
                outs.append("")
        decode_elapsed = time.perf_counter() - t0

        n_ok = 0
        failures: List[Tuple[str, str]] = []
        for w, out in zip(words, outs):
            if out == w:
                n_ok += 1
            elif len(failures) < 200:
                failures.append((w, out))

        acc = n_ok / max(n_total, 1)
        rows.append({
            "tokenizer": name,
            "n_words": n_total,
            "roundtrip_acc": round(acc, 4),
            "roundtrip_fail_pct": round((1 - acc) * 100, 2),
            "n_fail": n_total - n_ok,
            "decode_words_per_sec": round(n_total / decode_elapsed, 1) if decode_elapsed > 0 else 0.0,
        })

        prefix = "roundtrip" if label == "words" else f"roundtrip_{label}"
        fail_path = output_dir / f"{prefix}_fail_{name}.csv"
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wordform", "reconstructed"])
            writer.writerows(failures)

        global_logger.info(
            f"[roundtrip_eval] {name}: acc={acc:.4f} fail={n_total - n_ok}/{n_total}"
        )

    rows.sort(key=lambda r: -r["roundtrip_acc"])
    prefix = "roundtrip" if label == "words" else f"roundtrip_{label}"
    summary_path = output_dir / f"{prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"ROUNDTRIP RECONSTRUCTION  (decode(encode(x)) == x, {label}, strip={strip})")
    print("=" * 70)
    print(f"  {'tokenizer':<20s} {'n_words':>8s} {'acc':>8s} {'fail%':>7s} {'decode w/s':>11s}")
    for r in rows:
        print(
            f"  {r['tokenizer']:<20s} {r['n_words']:>8,} {r['roundtrip_acc']:>8.4f} "
            f"{r['roundtrip_fail_pct']:>7.2f} {r['decode_words_per_sec']:>11,.0f}"
        )
    print("=" * 70)
    print(f"Summary CSV: {summary_path}")
    return rows


def load_corpus_lines(corpus_path: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    with open(corpus_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break
    return lines


def run(
        classical_vocab: int = 64000,
        mode: str = "all",
        max_sentences: int = 5000,
) -> List[Dict]:
    base = Path(__file__).resolve().parents[3]
    artifacts = base / "src" / "model_development" / "artifacts"
    data_path = base / "data" / "morphscore" / "turkish_data.csv"
    corpus_path = artifacts / "datasets" / "splits" / "test.txt"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "roundtrip"

    codecs = build_codecs(artifacts, classical_vocab)
    if not codecs:
        global_logger.error("[roundtrip_eval] No tokenizers available.")
        return []

    rows: List[Dict] = []
    if mode in ("words", "all"):
        df = load_morphscore_turkish(data_path)
        words = df["wordform"].astype(str).tolist()
        global_logger.info(f"[roundtrip_eval] {len(words):,} inflected wordforms")
        rows += evaluate(words, codecs, output_dir, label="words", strip=True)

    if mode in ("sentences", "all"):
        if corpus_path.exists():
            sentences = load_corpus_lines(corpus_path, max_sentences)
            global_logger.info(f"[roundtrip_eval] {len(sentences):,} raw corpus lines from {corpus_path}")
            rows += evaluate(sentences, codecs, output_dir, label="sentences", strip=False)
        else:
            global_logger.warning(f"[roundtrip_eval] {corpus_path} missing — skipping corpus sentences")
        rows += evaluate(STRESS_SENTENCES, codecs, output_dir, label="stress", strip=False)

    return rows


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.benchmarks.roundtrip_eval")
    parser.add_argument("--classical-vocab", type=int, default=64000)
    parser.add_argument("--mode", choices=["words", "sentences", "all"], default="all")
    parser.add_argument("--max-sentences", type=int, default=5000)
    args = parser.parse_args()
    rows = run(
        classical_vocab=args.classical_vocab,
        mode=args.mode,
        max_sentences=args.max_sentences,
    )
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
