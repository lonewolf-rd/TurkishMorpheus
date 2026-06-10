import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Union

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger


LEXICON_URL = (
    "https://raw.githubusercontent.com/malibayram/tokenizer_benchmark/"
    "main/veri/turkish_ekler_kokler.txt"
)

_SUBWORD_MARKERS = ("▁", "##", "Ġ", "Ċ")


TURKISH_AFFIXES: frozenset = frozenset({
    "lar", "ler",
    "ı", "i", "u", "ü", "yı", "yi", "yu", "yü",
    "a", "e", "ya", "ye", "na", "ne",
    "da", "de", "ta", "te", "nda", "nde",
    "dan", "den", "tan", "ten", "ndan", "nden",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    "la", "le", "yla", "yle",
    "m", "ım", "im", "um", "üm",
    "n", "nı", "ni", "nu", "nü",
    "sı", "si", "su", "sü",
    "mız", "miz", "muz", "müz", "ımız", "imiz", "umuz", "ümüz",
    "nız", "niz", "nuz", "nüz", "ınız", "iniz", "unuz", "ünüz",
    "ları", "leri",
    "ki", "deki", "daki", "teki", "taki", "ndaki", "ndeki",
    "yım", "yim", "yum", "yüm",
    "sın", "sin", "sun", "sün",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz",
    "sınız", "siniz", "sunuz", "sünüz",
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
    "ydı", "ydi", "ydu", "ydü",
    "mış", "miş", "muş", "müş", "ymış", "ymiş", "ymuş", "ymüş",
    "sa", "se", "ysa", "yse",
    "ken", "yken", "iken",
    "yor", "ıyor", "iyor", "uyor", "üyor",
    "acak", "ecek", "yacak", "yecek",
    "ar", "er", "ır", "ir", "ur", "ür", "r",
    "maz", "mez",
    "ma", "me",
    "mı", "mi", "mu", "mü",
    "malı", "meli",
    "abil", "ebil", "yabil", "yebil",
    "mak", "mek",
    "ış", "iş", "uş", "üş", "yış", "yiş",
    "an", "en", "yan", "yen",
    "dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük",
    "arak", "erek", "yarak", "yerek",
    "ıp", "ip", "up", "üp", "yıp", "yip", "yup", "yüp",
    "ınca", "ince", "unca", "ünce", "yınca", "yince", "yunca", "yünce",
    "dıkça", "dikçe", "dukça", "dükçe", "tıkça", "tikçe", "tukça", "tükçe",
    "madan", "meden",
    "ıl", "il", "ul", "ül",
    "t", "ıt", "it", "ut", "üt",
    "lı", "li", "lu", "lü",
    "sız", "siz", "suz", "süz",
    "lık", "lik", "luk", "lük",
    "cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü",
    "cık", "cik", "cuk", "cük", "çık", "çik", "çuk", "çük",
    "ca", "ce", "ça", "çe",
    "daş", "deş", "taş", "teş",
    "ncı", "nci", "ncu", "ncü", "ıncı", "inci", "uncu", "üncü",
    "sal", "sel",
    "msı", "msi", "ımsı", "imsi",
    "lan", "len", "laş", "leş",
    "gan", "gen", "kan",
    "gın", "gin", "gun", "gün", "kın", "kin", "kun", "kün",
})


def clean_token(token: str) -> str:
    t = token
    for marker in _SUBWORD_MARKERS:
        t = t.replace(marker, "")
    return t.strip()


def ensure_lexicon(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    global_logger.info(
        f"[TurkishLexicalValidator](ensure_lexicon) Lexicon missing, "
        f"downloading from {LEXICON_URL} -> {path}"
    )
    urllib.request.urlretrieve(LEXICON_URL, str(path))
    return path


class TurkishLexicalValidator:
    def __init__(
            self,
            lexicon_path: Union[str, Path],
            use_analyzer: bool = True,
            include_affixes: bool = True,
    ):
        self.affixes: frozenset = TURKISH_AFFIXES if include_affixes else frozenset()
        self.lexicon: Set[str] = set()
        path = ensure_lexicon(lexicon_path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if entry:
                    self.lexicon.add(entry)
                    self.lexicon.add(turkish_lower(entry))
        global_logger.info(
            f"[TurkishLexicalValidator](__init__) Lexicon loaded: "
            f"{len(self.lexicon):,} entries from {path}"
        )

        self.analyzer = None
        self.analyzer_name = "lexicon-only"
        if use_analyzer:
            try:
                import zeyrek
                self.analyzer = zeyrek.MorphAnalyzer()
                self.analyzer_name = "zeyrek"
                global_logger.info(
                    "[TurkishLexicalValidator](__init__) zeyrek morphological analyzer active"
                )
            except ImportError:
                global_logger.warning(
                    "[TurkishLexicalValidator](__init__) zeyrek not installed — "
                    "%TR falls back to lexicon-only matching (pip install zeyrek)"
                )

        self._analysis_cache: Dict[str, bool] = {}

    def is_pure(self, token: str) -> bool:
        t = clean_token(token)
        if not t:
            return False
        if t in self.lexicon or turkish_lower(t) in self.lexicon:
            return True
        return turkish_lower(t) in self.affixes

    def _analyzer_accepts(self, word: str) -> bool:
        cached = self._analysis_cache.get(word)
        if cached is not None:
            return cached
        ok = False
        if self.analyzer is not None and word:
            try:
                parses = self.analyzer.analyze(word)
                ok = any(bool(word_parses) for word_parses in parses)
            except Exception:
                ok = False
        self._analysis_cache[word] = ok
        return ok

    def is_turkish(self, token: str) -> bool:
        t = clean_token(token)
        if not t:
            return False
        if self.is_pure(t):
            return True
        return self._analyzer_accepts(turkish_lower(t))

    def classify(self, tokens: Iterable[str]) -> Dict[str, Tuple[bool, bool]]:
        results: Dict[str, Tuple[bool, bool]] = {}
        for tok in tokens:
            pure = self.is_pure(tok)
            turkish = True if pure else self.is_turkish(tok)
            results[tok] = (turkish, pure)
        return results
