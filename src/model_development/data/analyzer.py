import os
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from src.common.providers.logger_provider import md_logger as global_logger
from src.common.providers.config_provider import md_config as config_provider


class DatasetAnalyzer:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    _RAW_FILENAME = "corpus.txt"

    def __init__(self):
        self.logger = global_logger
        self.configs = config_provider.cfg

        artifacts_dir = self._PROJECT_ROOT / "src/model_development/artifacts/datasets"
        self.input_file = artifacts_dir / "raw" / self._RAW_FILENAME
        self.output_dir = artifacts_dir / "splits"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_split(self):
        self.logger.info(f"[DatasetAnalyzer](analyze_and_split) Starting comprehensive analysis for {self.input_file.name}")

        if not self.input_file.exists():
            self.logger.error(f"[DatasetAnalyzer](analyze_and_split) Source file not found: {self.input_file}")
            return

        with open(self.input_file, "r", encoding="utf-8") as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        full_text = " ".join(raw_lines)
        all_words = full_text.split()

        stats = {
            "line_count": len(raw_lines),
            "word_count": len(all_words),
            "char_count": len(full_text),
            "unique_words": len(set(all_words)),
            "file_size_mb": os.path.getsize(self.input_file) / (1024 * 1024)
        }

        self._print_quantitative_stats(stats)
        tr_chars = "çğışöüÇĞİŞÖÜ"
        char_counts = Counter(full_text)
        tr_char_freq = {c: char_counts.get(c, 0) for c in tr_chars if char_counts.get(c, 0) > 0}
        self._print_char_frequencies(tr_char_freq)

        word_lengths = [len(w) for w in all_words]
        avg_len = sum(word_lengths) / max(stats["word_count"], 1)
        max_word = max(all_words, key=len) if all_words else ""

        self.logger.info("\n" + "=" * 55)
        self.logger.info("MORPHOLOGICAL DENSITY (WORD LENGTHS)")
        self.logger.info("=" * 55)
        self.logger.info(f"  Average Length    : {avg_len:.2f} chars")
        self.logger.info(f"  Max Word Length   : {len(max_word)} chars")
        self.logger.info(f"  Sample Max Word   : {max_word[:50]}...")
        self.logger.info("=" * 55)

        self._create_splits(raw_lines)

    def _print_quantitative_stats(self, stats: Dict[str, Any]):
        self.logger.info("\n" + "=" * 55)
        self.logger.info("CORPUS QUANTITATIVE STATISTICS")
        self.logger.info("=" * 55)
        self.logger.info(f"  Line Count        : {stats['line_count']:>15,}")
        self.logger.info(f"  Word Count        : {stats['word_count']:>15,}")
        self.logger.info(f"  Character Count   : {stats['char_count']:>15,}")
        self.logger.info(f"  Unique Vocabulary : {stats['unique_words']:>15,}")
        self.logger.info(f"  Total File Size   : {stats['file_size_mb']:>12.2f} MB")
        self.logger.info("=" * 55)

    def _print_char_frequencies(self, tr_char_freq: Dict[str, int]):
        if not tr_char_freq:
            return
        self.logger.info("[DatasetAnalyzer] Turkish Character Frequencies")
        max_val = max(tr_char_freq.values())
        for c, n in sorted(tr_char_freq.items(), key=lambda x: -x[1]):
            bar_len = int((n / max_val) * 30) + 1
            bar = "#" * bar_len
            self.logger.info(f"  {c}: {n:>10,} {bar}")

    def _create_splits(self, lines: List[str]):
        random.seed(self.configs.dataset.get("seed", 42))
        random.shuffle(lines)

        split_ratio = self.configs.dataset.get("train_split_ratio", 0.95)
        split_idx = int(len(lines) * split_ratio)

        train_lines = lines[:split_idx]
        test_lines = lines[split_idx:]

        train_file = self.output_dir / "train.txt"
        test_file = self.output_dir / "test.txt"

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_lines))

        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines))

        self.logger.info(f"[DatasetAnalyzer](_create_splits) Saved {len(train_lines)} lines to train.txt")
        self.logger.info(f"[DatasetAnalyzer](_create_splits) Saved {len(test_lines)} lines to test.txt")


if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.analyze_and_split()
