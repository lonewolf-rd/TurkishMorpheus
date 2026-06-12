import urllib.request
from pathlib import Path
from typing import Optional, Set

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger


KALBUR_BASE = "https://raw.githubusercontent.com/ahmetax/kalbur/master/veri/"
KALBUR_ROOT_FILES = ("KOKLER.txt", "KOKOZLER.txt")
KALBUR_SUFFIX_FILE = "EKLER.txt"
_SOFTENING = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}


def _default_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "tr_lexicon" / "kalbur"


def _ensure(filename: str, data_dir: Path) -> Path:
    path = data_dir / filename
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    global_logger.info(f"[KalburRoots] downloading {KALBUR_BASE + filename} -> {path}")
    urllib.request.urlretrieve(KALBUR_BASE + filename, str(path))
    return path


def _load_first_column(path: Path) -> Set[str]:
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                out.add(turkish_lower(parts[0]))
    return out


class KalburRoots:
    def __init__(self, data_dir: Optional[Path] = None):
        data_dir = Path(data_dir) if data_dir else _default_dir()
        self.roots: Set[str] = set()
        self.suffixes: Set[str] = set()
        for fname in KALBUR_ROOT_FILES:
            self.roots |= _load_first_column(_ensure(fname, data_dir))
        self.suffixes |= _load_first_column(_ensure(KALBUR_SUFFIX_FILE, data_dir))
        global_logger.info(
            f"[KalburRoots] {len(self.roots):,} roots, {len(self.suffixes):,} suffixes "
            f"from {data_dir}"
        )

    def _root_match(self, t: str) -> bool:
        if t in self.roots:
            return True
        if t and t[-1] in _SOFTENING:
            hardened = t[:-1] + _SOFTENING[t[-1]]
            if hardened in self.roots:
                return True
        return False

    def root_split(self, word: str) -> Optional[int]:
        t = turkish_lower(word)
        n = len(t)
        for length in range(n, 0, -1):
            root = t[:length]
            suffix = t[length:]
            if self._root_match(root) and (suffix == "" or suffix in self.suffixes):
                return length
        return None
