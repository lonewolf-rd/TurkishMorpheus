import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger


LEXICON_URL = (
    "https://raw.githubusercontent.com/malibayram/tokenizer_benchmark/"
    "main/veri/turkish_ekler_kokler.txt"
)

_SUBWORD_MARKERS = ("▁", "##", "Ġ", "Ċ")

_SOFTENING = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}


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


def quiet_zeyrek_logging() -> None:
    import logging
    for name in list(logging.root.manager.loggerDict.keys()) + ["zeyrek"]:
        if name == "zeyrek" or name.startswith("zeyrek."):
            logging.getLogger(name).setLevel(logging.ERROR)


def ensure_nltk_data() -> None:
    try:
        import nltk
    except ImportError:
        global_logger.warning(
            "[ensure_nltk_data] nltk not importable — zeyrek will fail to tokenize"
        )
        return
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            global_logger.warning(
                f"[ensure_nltk_data] nltk.download('{resource}') failed: "
                f"{type(e).__name__}: {e}"
            )


class TurkishLexicalValidator:
    def __init__(
            self,
            lexicon_path: Union[str, Path],
            use_analyzer: bool = True,
            include_affixes: bool = True,
            inventory_path: Optional[Union[str, Path]] = None,
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

        self.harvested_morphemes: Set[str] = set()
        self.harvested_words: Set[str] = set()
        inv_path = Path(inventory_path) if inventory_path else (
            Path(path).parent / "harvested_morphemes.json"
        )
        if inv_path.exists():
            with open(inv_path, "r", encoding="utf-8") as f:
                inv = json.load(f)
            self.harvested_morphemes = {turkish_lower(m) for m in inv.get("morphemes", [])}
            self.harvested_words = {turkish_lower(w) for w in inv.get("words", [])}
            global_logger.info(
                f"[TurkishLexicalValidator](__init__) Harvested inventory: "
                f"{len(self.harvested_morphemes):,} morphemes, "
                f"{len(self.harvested_words):,} words from {inv_path}"
            )

        self.analyzer = None
        self.analyzer_name = "lexicon-only"
        self._analyzer_error_logged = False
        self._analysis_cache: Dict[str, bool] = {}

        if self.harvested_words:
            use_analyzer = False
            self.analyzer_name = "harvested-offline"

        if use_analyzer:
            try:
                import zeyrek
                ensure_nltk_data()
                self.analyzer = zeyrek.MorphAnalyzer()
                if self._analyzer_selftest():
                    self.analyzer_name = "zeyrek"
                    global_logger.info(
                        "[TurkishLexicalValidator](__init__) zeyrek morphological analyzer active"
                    )
                else:
                    self.analyzer = None
                    global_logger.error(
                        "[TurkishLexicalValidator](__init__) zeyrek imported but self-test "
                        "FAILED — disabling analyzer; %TR will equal %Pure. "
                        "Fix: python -c \"import nltk; nltk.download('punkt_tab')\""
                    )
            except ImportError:
                global_logger.warning(
                    "[TurkishLexicalValidator](__init__) zeyrek not installed — "
                    "%TR falls back to lexicon-only matching (pip install zeyrek)"
                )

    def _analyzer_selftest(self) -> bool:
        probes = ["gidiyor", "evlerimizde", "kitap"]
        n_ok = 0
        for w in probes:
            try:
                parses = self.analyzer.analyze(w)
                if any(len(wp) > 0 for wp in parses):
                    n_ok += 1
            except Exception as e:
                global_logger.error(
                    f"[TurkishLexicalValidator](_analyzer_selftest) zeyrek.analyze('{w}') "
                    f"raised {type(e).__name__}: {str(e).splitlines()[0]}"
                )
                return False
        global_logger.info(
            f"[TurkishLexicalValidator](_analyzer_selftest) zeyrek parsed "
            f"{n_ok}/{len(probes)} probe words"
        )
        return n_ok > 0

    def _lexicon_has(self, token: str) -> bool:
        if token in self.lexicon:
            return True
        tl = turkish_lower(token)
        if tl in self.lexicon:
            return True
        if tl and tl[-1] in _SOFTENING:
            hardened = tl[:-1] + _SOFTENING[tl[-1]]
            if hardened in self.lexicon:
                return True
        return False

    def is_pure(self, token: str) -> bool:
        t = clean_token(token)
        if not t:
            return False
        if self._lexicon_has(t):
            return True
        tl = turkish_lower(t)
        return tl in self.affixes or tl in self.harvested_morphemes

    def _analyzer_accepts(self, word: str) -> bool:
        cached = self._analysis_cache.get(word)
        if cached is not None:
            return cached
        ok = False
        if self.analyzer is not None and word:
            try:
                parses = self.analyzer.analyze(word)
                for word_parses in parses:
                    for parse in word_parses:
                        pos = getattr(parse, "pos", None)
                        if pos is None:
                            ok = bool(parse)
                        elif str(pos).lower() not in ("unk", "unknown", "punc"):
                            ok = True
                        if ok:
                            break
                    if ok:
                        break
            except Exception as e:
                ok = False
                if not self._analyzer_error_logged:
                    self._analyzer_error_logged = True
                    global_logger.error(
                        f"[TurkishLexicalValidator](_analyzer_accepts) zeyrek.analyze raised "
                        f"{type(e).__name__}: {str(e).splitlines()[0]} "
                        f"(further errors suppressed; %TR may be understated)"
                    )
        self._analysis_cache[word] = ok
        return ok

    def is_turkish(self, token: str) -> bool:
        t = clean_token(token)
        if not t:
            return False
        if self.is_pure(t):
            return True
        tl = turkish_lower(t)
        return tl in self.harvested_words or self._analyzer_accepts(tl)

    def classify(self, tokens: Iterable[str]) -> Dict[str, Tuple[bool, bool]]:
        results: Dict[str, Tuple[bool, bool]] = {}
        rescued = 0
        for tok in tokens:
            pure = self.is_pure(tok)
            if pure:
                turkish = True
            else:
                tl = turkish_lower(clean_token(tok))
                turkish = tl in self.harvested_words or self._analyzer_accepts(tl)
                if turkish:
                    rescued += 1
            results[tok] = (turkish, pure)
        global_logger.info(
            f"[TurkishLexicalValidator](classify) {len(results):,} unique tokens, "
            f"validator ({self.analyzer_name}) credited {rescued:,} non-pure tokens as Turkish"
        )
        return results
