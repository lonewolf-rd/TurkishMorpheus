import sys
import time
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger
from src.benchmarker.metrics.linguistic_validity import TurkishLexicalValidator
from src.benchmarker.metrics.efficiency import renyi_efficiency
from src.benchmarker.benchmarks.paper import (
    ClassicalTokenizerWrapper,
    discover_classical_tokenizers,
)


TRMMLU_PARQUET_URL = (
    "https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu_model_cevaplari/"
    "resolve/main/data/train-00000-of-00001.parquet"
)

MORPHEUS_PREFERRED_CHECKPOINTS = [
    "turkish_morpheus_a100_v3_best.pt",
    "turkish_morpheus_a100_release_best.pt",
    "turkish_morpheus_a100_best.pt",
]


def load_trmmlu_text(cache_path: Path) -> str:
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        global_logger.info(
            f"[trmmlu_eval](load_trmmlu_text) Loaded cached TR-MMLU text: "
            f"{len(text):,} chars"
        )
        return text

    global_logger.info(
        f"[trmmlu_eval](load_trmmlu_text) Downloading TR-MMLU parquet from "
        f"{TRMMLU_PARQUET_URL}"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_path.parent / "trmmlu_source.parquet"
    if not parquet_path.exists():
        import urllib.request
        urllib.request.urlretrieve(TRMMLU_PARQUET_URL, str(parquet_path))
    df = pd.read_parquet(parquet_path)
    parts: List[str] = []
    for _, row in df.iterrows():
        parts.append(str(row["soru"]))
        for secenek in row["secenekler"]:
            parts.append(str(secenek))
    text = "\n".join(parts) + "\n"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    n_words = len(text.split())
    global_logger.info(
        f"[trmmlu_eval](load_trmmlu_text) Built TR-MMLU text: "
        f"{len(df):,} questions, {len(text):,} chars, {n_words:,} words -> {cache_path}"
    )
    return text


class PieceAdapter:
    name: str = "base"
    vocab_size: Optional[int] = None

    def pieces(self, lines: List[str]) -> List[str]:
        raise NotImplementedError


class ClassicalPieceAdapter(PieceAdapter):
    def __init__(self, wrapper: ClassicalTokenizerWrapper):
        self.wrapper = wrapper
        self.name = wrapper.name.lower()
        self.vocab_size = wrapper.vocab_size

    def pieces(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        if self.wrapper.kind in ("bpe", "byte_bpe", "unigram"):
            for line in lines:
                out.extend(self.wrapper.model.encode_as_pieces(line))
        else:
            specials = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}
            for line in lines:
                out.extend(
                    t for t in self.wrapper.model.encode(line).tokens
                    if t not in specials
                )
        return out


class MorfessorPieceAdapter(PieceAdapter):
    def __init__(self, model_path: str):
        from src.model_development.training.dataset import MorfessorWrapper
        self.name = "morfessor"
        self.vocab_size = None
        self.wrapper = MorfessorWrapper(model_path)
        self._cache: Dict[str, List[str]] = {}

    def _segment(self, word: str) -> List[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        segs, _ = self.wrapper.segment(word)
        self._cache[word] = segs
        return segs

    def pieces(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            for word in line.split():
                out.extend(self._segment(word))
        return out


class MorpheusPieceAdapter(PieceAdapter):
    def __init__(self, checkpoint_path: str, tokenizer_dir: str):
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

        self.tokenizer = MorpheusTokenizer.load(
            tokenizer_dir,
            morpheus_model=model,
            device=device,
        )
        self.name = "morpheus"
        self.vocab_size = self.tokenizer.vocab_size

    def pieces(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            out.extend(self.tokenizer.tokenize(line))
        return out


class HFTokenizerAdapter(PieceAdapter):
    def __init__(self, name: str, model_id: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.name = name
        self.vocab_size = getattr(self.tok, "vocab_size", None)

    def pieces(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            for tid in self.tok.encode(line, add_special_tokens=False):
                out.append(self.tok.decode([tid]))
        return out


class TiktokenAdapter(PieceAdapter):
    def __init__(self, name: str, encoding_name: str):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.name = name
        self.vocab_size = self.enc.n_vocab

    def pieces(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        decode = self.enc.decode
        for line in lines:
            for tid in self.enc.encode(line):
                out.append(decode([tid]))
        return out


def build_reference_adapters(
        reference_hf: Optional[str],
        reference_tiktoken: Optional[str],
) -> List[PieceAdapter]:
    adapters: List[PieceAdapter] = []

    if reference_hf:
        for spec in reference_hf.split(","):
            spec = spec.strip()
            if not spec:
                continue
            model_id, _, name = spec.partition("=")
            name = name or model_id.split("/")[-1]
            try:
                adapters.append(HFTokenizerAdapter(name, model_id))
                global_logger.info(f"[trmmlu_eval](reference) loaded HF tokenizer {model_id} as '{name}'")
            except Exception as e:
                global_logger.error(f"[trmmlu_eval](reference) HF {model_id} failed: {e}")

    if reference_tiktoken:
        for spec in reference_tiktoken.split(","):
            spec = spec.strip()
            if not spec:
                continue
            encoding, _, name = spec.partition("=")
            name = name or encoding
            try:
                adapters.append(TiktokenAdapter(name, encoding))
                global_logger.info(f"[trmmlu_eval](reference) loaded tiktoken {encoding} as '{name}'")
            except Exception as e:
                global_logger.error(f"[trmmlu_eval](reference) tiktoken {encoding} failed: {e}")

    return adapters


def find_morpheus_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    for cand in MORPHEUS_PREFERRED_CHECKPOINTS:
        p = checkpoints_dir / cand
        if p.exists():
            return p
    if checkpoints_dir.exists():
        best_pts = sorted(
            checkpoints_dir.glob("*_best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if best_pts:
            return best_pts[0]
    return None


def build_adapters(
        artifacts_dir: Path,
        preferred_classical_vocab: int,
) -> List[PieceAdapter]:
    adapters: List[PieceAdapter] = []
    classical_dir = artifacts_dir / "tokenizers" / "classical"
    morpheus_dir = artifacts_dir / "tokenizers" / "morpheus_50k"
    checkpoints_dir = artifacts_dir / "checkpoints"

    for wrapper in discover_classical_tokenizers(
            str(classical_dir), preferred_classical_vocab):
        adapters.append(ClassicalPieceAdapter(wrapper))

    morfessor_path = classical_dir / "morfessor_model.bin"
    if morfessor_path.exists():
        adapters.append(MorfessorPieceAdapter(str(morfessor_path)))
    else:
        global_logger.warning(
            f"[trmmlu_eval](build_adapters) morfessor_model.bin not found at {morfessor_path}"
        )

    ckpt = find_morpheus_checkpoint(checkpoints_dir)
    if ckpt and morpheus_dir.exists():
        adapters.append(MorpheusPieceAdapter(str(ckpt), str(morpheus_dir)))
    else:
        global_logger.warning(
            f"[trmmlu_eval](build_adapters) Morpheus tokenizer missing "
            f"(ckpt={ckpt}, tokenizer_dir_exists={morpheus_dir.exists()})"
        )

    return adapters


def evaluate_adapter(
        adapter: PieceAdapter,
        lines: List[str],
        n_words: int,
        validator: TurkishLexicalValidator,
        output_dir: Path,
) -> Dict:
    global_logger.info(f"[trmmlu_eval](evaluate_adapter) === {adapter.name} ===")

    t0 = time.perf_counter()
    tokens = adapter.pieces(lines)
    elapsed = time.perf_counter() - t0

    counts = Counter(tokens)
    n_total = len(tokens)
    n_unique = len(counts)

    classification = validator.classify(counts.keys())

    n_turkish_unique = sum(1 for t in counts if classification[t][0])
    n_pure_unique = sum(1 for t in counts if classification[t][1])
    n_turkish_weighted = sum(c for t, c in counts.items() if classification[t][0])
    n_pure_weighted = sum(c for t, c in counts.items() if classification[t][1])

    efficiency = renyi_efficiency(counts, alpha=2.5, vocab_size=adapter.vocab_size)

    audit_path = output_dir / f"trmmlu_tokens_{adapter.name}.csv"
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count", "is_turkish", "is_pure"])
        for tok, c in counts.most_common():
            turkish, pure = classification[tok]
            writer.writerow([tok, c, int(turkish), int(pure)])

    row = {
        "tokenizer": adapter.name,
        "vocab_size": adapter.vocab_size if adapter.vocab_size else "",
        "total_tokens": n_total,
        "unique_tokens": n_unique,
        "process_time_s": round(elapsed, 4),
        "tr_pct": round(n_turkish_unique / max(n_unique, 1) * 100, 2),
        "pure_pct": round(n_pure_unique / max(n_unique, 1) * 100, 2),
        "tr_pct_freq_weighted": round(n_turkish_weighted / max(n_total, 1) * 100, 2),
        "pure_pct_freq_weighted": round(n_pure_weighted / max(n_total, 1) * 100, 2),
        "fertility_tokens_per_word": round(n_total / max(n_words, 1), 4),
        **efficiency,
        "validator": validator.analyzer_name,
    }

    global_logger.info(
        f"[trmmlu_eval](evaluate_adapter) {adapter.name}: "
        f"tokens={n_total:,} unique={n_unique:,} "
        f"TR%={row['tr_pct']} Pure%={row['pure_pct']} "
        f"renyi_eff={efficiency['renyi_efficiency_observed']} "
        f"time={elapsed:.2f}s"
    )
    return row


def build_validator(validator_kind: str, base: Path, use_analyzer: bool):
    if validator_kind == "kalbur":
        from src.benchmarker.metrics.kalbur_validator import KalburValidator
        return KalburValidator(base / "data" / "tr_lexicon" / "kalbur")
    lexicon_path = base / "data" / "tr_lexicon" / "turkish_ekler_kokler.txt"
    return TurkishLexicalValidator(lexicon_path, use_analyzer=use_analyzer)


def run(
        classical_vocab: int = 64000,
        tokenizers: Optional[str] = None,
        use_analyzer: bool = True,
        validator_kind: str = "kalbur",
        reference_hf: Optional[str] = None,
        reference_tiktoken: Optional[str] = None,
) -> List[Dict]:
    base = Path(__file__).resolve().parents[3]
    artifacts = base / "src" / "model_development" / "artifacts"
    text_cache = base / "data" / "trmmlu" / "trmmlu_text.txt"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "trmmlu"
    output_dir.mkdir(parents=True, exist_ok=True)

    text = load_trmmlu_text(text_cache)
    raw_lines = [line for line in text.splitlines() if line.strip()]
    lines = [turkish_lower(line) for line in raw_lines]
    n_words = sum(len(line.split()) for line in lines)
    n_chars = sum(len(line) for line in lines)
    global_logger.info(
        f"[trmmlu_eval](run) Evaluation text: {len(lines):,} lines, "
        f"{n_words:,} words, {n_chars:,} chars"
    )

    validator = build_validator(validator_kind, base, use_analyzer)

    adapters = build_adapters(artifacts, classical_vocab)
    if tokenizers:
        wanted = {s.strip() for s in tokenizers.split(",")}
        adapters = [a for a in adapters if a.name in wanted]

    reference_adapters = build_reference_adapters(reference_hf, reference_tiktoken)

    global_logger.info(
        f"[trmmlu_eval](run) adapters: {[a.name for a in adapters]} "
        f"| references: {[a.name for a in reference_adapters]}"
    )

    rows: List[Dict] = []
    for adapter in adapters:
        try:
            rows.append(evaluate_adapter(adapter, lines, n_words, validator, output_dir))
        except Exception as e:
            global_logger.error(f"[trmmlu_eval](run) {adapter.name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    for adapter in reference_adapters:
        try:
            row = evaluate_adapter(adapter, raw_lines, n_words, validator, output_dir)
            row["is_reference"] = True
            rows.append(row)
        except Exception as e:
            global_logger.error(f"[trmmlu_eval](run) reference {adapter.name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not rows:
        global_logger.error("[trmmlu_eval](run) No results produced.")
        return rows

    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    summary_path = output_dir / "trmmlu_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 88)
    print("TR-MMLU TOKENIZER EVALUATION  (%TR / %Pure over unique tokens — Bayram et al. protocol)")
    print("=" * 88)
    print(
        f"  {'tokenizer':<18s} {'vocab':>7s} {'tokens':>10s} {'unique':>8s} "
        f"{'TR%':>7s} {'Pure%':>7s} {'RenyiEff':>9s} {'time(s)':>8s}"
    )
    for r in sorted(rows, key=lambda r: -r["tr_pct"]):
        vocab_str = f"{r['vocab_size']:,}" if r["vocab_size"] else "n/a"
        marker = " [ref]" if r.get("is_reference") else ""
        print(
            f"  {r['tokenizer']:<18s} {vocab_str:>7s} {r['total_tokens']:>10,} "
            f"{r['unique_tokens']:>8,} {r['tr_pct']:>7.2f} {r['pure_pct']:>7.2f} "
            f"{r['renyi_efficiency_observed']:>9.4f} {r['process_time_s']:>8.2f}{marker}"
        )
    print("=" * 88)
    print(f"Validator: {validator.analyzer_name}")
    print(f"Summary CSV: {summary_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.benchmarks.trmmlu_eval")
    parser.add_argument("--classical-vocab", type=int, default=64000)
    parser.add_argument("--tokenizers", default=None,
                        help="Comma-separated subset of tokenizer names to run (default: all)")
    parser.add_argument("--no-analyzer", action="store_true",
                        help="Disable zeyrek fallback; %%TR uses lexicon-only matching")
    parser.add_argument("--validator", choices=["kalbur", "lexical"], default="kalbur",
                        help="kalbur: offline KOKLER/EKLER root+suffix (faithful to Bayram). "
                             "lexical: turkish_ekler_kokler + harvested/zeyrek.")
    parser.add_argument("--reference-hf", default=None,
                        help="Comma-separated HF tokenizer ids for calibration, "
                             "e.g. 'Qwen/Qwen2.5-7B=qwen2.5,google/gemma-2-9b=gemma-2'")
    parser.add_argument("--reference-tiktoken", default=None,
                        help="Comma-separated tiktoken encodings, e.g. 'o200k_base=gpt-4o'")
    args = parser.parse_args()

    rows = run(
        classical_vocab=args.classical_vocab,
        tokenizers=args.tokenizers,
        use_analyzer=not args.no_analyzer,
        validator_kind=args.validator,
        reference_hf=args.reference_hf,
        reference_tiktoken=args.reference_tiktoken,
    )
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
