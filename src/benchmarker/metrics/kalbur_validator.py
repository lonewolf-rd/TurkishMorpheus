import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Union

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger


KALBUR_BASE = "https://raw.githubusercontent.com/ahmetax/kalbur/master/veri/"
KALBUR_ROOT_FILES = ("KOKLER.txt", "KOKOZLER.txt")
KALBUR_SUFFIX_FILE = "EKLER.txt"

_SUBWORD_MARKERS = ("▁", "##", "Ġ", "Ċ")
_SOFTENING = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}


def clean_token(token: str) -> str:
    t = token
    for marker in _SUBWORD_MARKERS:
        t = t.replace(marker, "")
    return t.strip()


def _ensure_file(filename: str, data_dir: Path) -> Path:
    path = data_dir / filename
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    url = KALBUR_BASE + filename
    global_logger.info(f"[KalburValidator](_ensure_file) downloading {url} -> {path}")
    urllib.request.urlretrieve(url, str(path))
    return path


def _load_first_column(path: Path) -> Set[str]:
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                out.add(turkish_lower(parts[0]))
    return out


class KalburValidator:
    analyzer_name = "kalbur"

    def __init__(self, data_dir: Union[str, Path]):
        data_dir = Path(data_dir)
        self.roots: Set[str] = set()
        self.suffixes: Set[str] = set()

        for fname in KALBUR_ROOT_FILES:
            self.roots |= _load_first_column(_ensure_file(fname, data_dir))
        self.suffixes |= _load_first_column(_ensure_file(KALBUR_SUFFIX_FILE, data_dir))

        global_logger.info(
            f"[KalburValidator](__init__) {len(self.roots):,} roots, "
            f"{len(self.suffixes):,} suffixes loaded from {data_dir}"
        )

    def _root_match(self, t: str) -> bool:
        if t in self.roots:
            return True
        if t and t[-1] in _SOFTENING:
            hardened = t[:-1] + _SOFTENING[t[-1]]
            if hardened in self.roots:
                return True
        return False

    def is_pure(self, token: str) -> bool:
        t = turkish_lower(clean_token(token))
        if not t:
            return False
        return self._root_match(t) or t in self.suffixes

    def _decomposes(self, t: str) -> bool:
        n = len(t)
        for split in range(n, 0, -1):
            root = t[:split]
            suffix = t[split:]
            if self._root_match(root) and (suffix == "" or suffix in self.suffixes):
                return True
        return False

    def is_turkish(self, token: str) -> bool:
        t = turkish_lower(clean_token(token))
        if not t:
            return False
        if self._root_match(t) or t in self.suffixes:
            return True
        return self._decomposes(t)

    def classify(self, tokens: Iterable[str]) -> Dict[str, Tuple[bool, bool]]:
        results: Dict[str, Tuple[bool, bool]] = {}
        rescued = 0
        for tok in tokens:
            pure = self.is_pure(tok)
            if pure:
                turkish = True
            else:
                turkish = self.is_turkish(tok)
                if turkish:
                    rescued += 1
            results[tok] = (turkish, pure)
        global_logger.info(
            f"[KalburValidator](classify) {len(results):,} unique tokens, "
            f"kalbur credited {rescued:,} non-pure tokens as Turkish (via root+suffix decomposition)"
        )
        return results
