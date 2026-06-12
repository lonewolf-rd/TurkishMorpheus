import os
import re
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Optional

from src.common.providers.logger_provider import global_logger


TR_ALPHABET = set("abcçdefgğhıijklmnoöprsştuüvyz")
NON_TR_LATIN = set("qwx")

TR_STOPWORDS = {
    "ve", "bir", "bu", "da", "de", "ki", "ile", "için", "çok", "daha", "en",
    "gibi", "ama", "ya", "veya", "ne", "şu", "her", "hem", "ise", "ancak",
    "çünkü", "kadar", "sonra", "önce", "olarak", "olan", "var", "yok", "değil",
    "mi", "mı", "mu", "mü", "ben", "sen", "biz", "siz", "onlar", "bütün",
    "olduğu", "olduğunu", "tarafından", "arasında", "üzerine", "göre", "değildir",
}

_WS = re.compile(r"\s+")
_MARKUP = re.compile(r"https?://|www\.|\[\[|\]\]|\{\{|\}\}|\||={2,}|&[a-z]+;|<[^>]+>")
_REF_BRACKET = re.compile(r"\[\d+\]|\[düzenle\]|\[kaynak belirtilmeli\]")


def _hash(line: str) -> str:
    return hashlib.md5(line.encode("utf-8")).hexdigest()[:16]


def clean_line(
        line: str,
        min_chars: int,
        max_chars: int,
        max_token_len: int,
        min_tr_ratio: float,
        max_nonalpha_ratio: float,
        max_foreign_ratio: float,
) -> Optional[str]:
    line = _REF_BRACKET.sub("", line)
    line = _WS.sub(" ", line).strip()

    if not (min_chars <= len(line) <= max_chars):
        return None
    if _MARKUP.search(line):
        return None

    tokens = line.split()
    if len(tokens) < 3:
        return None
    if any(len(t) > max_token_len for t in tokens):
        return None

    lower = line.lower()
    alpha = [c for c in lower if c.isalpha()]
    if not alpha:
        return None

    tr_alpha = sum(1 for c in alpha if c in TR_ALPHABET)
    if tr_alpha / len(alpha) < min_tr_ratio:
        return None

    foreign = sum(1 for c in alpha if c in NON_TR_LATIN)
    if foreign / len(alpha) > max_foreign_ratio:
        return None

    nonalpha = sum(1 for c in line if not c.isalpha() and not c.isspace())
    if nonalpha / max(len(line), 1) > max_nonalpha_ratio:
        return None

    words = set(_WS.sub(" ", lower).split())
    if words.isdisjoint(TR_STOPWORDS):
        return None

    return line


def ingest(
        output_path: Path,
        config: str,
        min_chars: int = 30,
        max_chars: int = 600,
        max_token_len: int = 34,
        min_tr_ratio: float = 0.92,
        max_nonalpha_ratio: float = 0.20,
        max_foreign_ratio: float = 0.02,
        limit: Optional[int] = None,
        append_to: Optional[Path] = None,
) -> Path:
    from datasets import load_dataset

    global_logger.info(f"[wikipedia_ingest] streaming wikimedia/wikipedia '{config}'")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    n_articles = 0
    n_lines_in = 0
    n_kept = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for article in ds:
            text = article.get("text", "")
            n_articles += 1
            for raw in text.split("\n"):
                n_lines_in += 1
                cleaned = clean_line(
                    raw, min_chars, max_chars, max_token_len,
                    min_tr_ratio, max_nonalpha_ratio, max_foreign_ratio,
                )
                if cleaned is None:
                    continue
                h = _hash(cleaned)
                if h in seen:
                    continue
                seen.add(h)
                out.write(cleaned + "\n")
                n_kept += 1

            if limit and n_articles >= limit:
                break
            if n_articles % 20000 == 0:
                global_logger.info(
                    f"[wikipedia_ingest] {n_articles:,} articles | "
                    f"{n_lines_in:,} lines seen | {n_kept:,} kept"
                )

    global_logger.info(
        f"[wikipedia_ingest] DONE: {n_articles:,} articles, "
        f"{n_lines_in:,} lines seen, {n_kept:,} clean unique lines -> {output_path} "
        f"(keep rate {n_kept / max(n_lines_in, 1) * 100:.1f}%)"
    )

    if append_to is not None:
        if append_to.exists():
            backup = append_to.with_suffix(append_to.suffix + ".bak")
            if not backup.exists():
                backup.write_bytes(append_to.read_bytes())
                global_logger.info(f"[wikipedia_ingest] backup -> {backup}")
        with open(append_to, "a", encoding="utf-8") as f:
            with open(output_path, "r", encoding="utf-8") as src:
                for line in src:
                    f.write(line)
        global_logger.info(f"[wikipedia_ingest] appended {n_kept:,} lines to {append_to}")

    return output_path


def main():
    base = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(prog="src.model_development.data.wikipedia_ingest")
    parser.add_argument("--config", default="20231101.tr")
    parser.add_argument("--out", default=str(base / "data/wikipedia_tr/wikipedia_tr_clean.txt"))
    parser.add_argument("--append-to", default=None,
                        help="Append cleaned lines to this corpus file (a .bak is made first), "
                             "e.g. data/corpus_collector_tr/corpus.txt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N articles (smoke test)")
    parser.add_argument("--min-tr-ratio", type=float, default=0.92)
    args = parser.parse_args()

    ingest(
        output_path=Path(args.out),
        config=args.config,
        min_tr_ratio=args.min_tr_ratio,
        limit=args.limit,
        append_to=Path(args.append_to) if args.append_to else None,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
