import json
import re
import argparse
from pathlib import Path
from typing import Iterable, Set

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger
from src.benchmarker.metrics.linguistic_validity import (
    ensure_nltk_data,
    quiet_zeyrek_logging,
)


_WORD_CLEAN = re.compile(r"[^\wçğıöşüÇĞİÖŞÜ]", re.UNICODE)
_SURFACE = re.compile(r"([a-zçğıöşü]+):")


def collect_words_from_text(path: Path) -> Set[str]:
    words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                w = _WORD_CLEAN.sub("", w)
                if 1 <= len(w) <= 40:
                    words.add(turkish_lower(w))
    return words


def collect_words_from_vocab(vocab_path: Path) -> Set[str]:
    import torch
    data = torch.load(str(vocab_path))
    vocab = data.get("vocab", data) if isinstance(data, dict) else {}
    return {turkish_lower(w) for w in vocab.keys()}


def build_inventory(
        word_sources: Iterable[Set[str]],
        output_path: Path,
        progress_every: int = 5000,
) -> None:
    ensure_nltk_data()
    quiet_zeyrek_logging()
    import zeyrek
    analyzer = zeyrek.MorphAnalyzer()

    words: Set[str] = set()
    for src in word_sources:
        words |= src
    global_logger.info(f"[build_inventory] {len(words):,} unique words to analyze")

    morphemes: Set[str] = set()
    valid_words: Set[str] = set()
    n_fail = 0

    for i, w in enumerate(sorted(words)):
        try:
            parses = analyzer.analyze(w)
        except Exception:
            n_fail += 1
            continue
        for word_parses in parses:
            for parse in word_parses:
                pos = getattr(parse, "pos", None)
                if pos and str(pos).lower() in ("unk", "unknown", "punc"):
                    continue
                valid_words.add(w)
                fmt = turkish_lower(getattr(parse, "formatted", "") or "")
                for surface in _SURFACE.findall(fmt):
                    morphemes.add(surface)
                lemma = getattr(parse, "lemma", None)
                if lemma:
                    morphemes.add(turkish_lower(lemma))
        if (i + 1) % progress_every == 0:
            global_logger.info(
                f"[build_inventory] {i + 1:,}/{len(words):,} | "
                f"morphemes={len(morphemes):,} words={len(valid_words):,}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"morphemes": sorted(morphemes), "words": sorted(valid_words)},
            f, ensure_ascii=False,
        )
    global_logger.info(
        f"[build_inventory] Saved {output_path}: "
        f"{len(morphemes):,} morphemes, {len(valid_words):,} words "
        f"({n_fail:,} unparseable)"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="src.benchmarker.metrics.build_morpheme_inventory"
    )
    parser.add_argument("--include-train-vocab", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[3]
    trmmlu_text = base / "data" / "trmmlu" / "trmmlu_text.txt"
    output = base / "data" / "tr_lexicon" / "harvested_morphemes.json"

    sources = []
    if trmmlu_text.exists():
        sources.append(collect_words_from_text(trmmlu_text))
    else:
        global_logger.error(
            f"[build_inventory] TR-MMLU text not found at {trmmlu_text} — "
            f"run trmmlu_eval once to download it first"
        )

    if args.include_train_vocab:
        vocab_path = base / "src/model_development/artifacts/datasets/splits/word_vocab.pt"
        if vocab_path.exists():
            sources.append(collect_words_from_vocab(vocab_path))
        else:
            global_logger.warning(
                f"[build_inventory] word_vocab.pt not found at {vocab_path}"
            )

    if not sources:
        raise SystemExit("[build_inventory] No word sources available.")

    build_inventory(sources, output)


if __name__ == "__main__":
    main()
