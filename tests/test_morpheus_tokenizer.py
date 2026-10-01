"""Model-free tests for MorpheusTokenizer's text <-> id plumbing.

The Morpheus segmenter is replaced by a deterministic stub, so these tests check
pre-tokenization, byte fallback, whitespace handling and decode only; they do not
need a checkpoint or a GPU.
"""
import pytest

from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.tokenization.morpheus_tokenizer import (
    BYTE_TOKENS,
    SPECIAL_TOKENS,
    WORD_BOUNDARY,
    MorpheusTokenizer,
    export_corpus_tokenized,
)


def _stub_segment(words):
    return [[w] if len(w) < 3 else [w[:2], w[2:]] for w in words]


@pytest.fixture
def tok():
    vocab = dict(SPECIAL_TOKENS)
    for ch in CharEncoderHelper._TURKISH_CHARS:
        if ch not in " \n\t" and ch not in vocab:
            vocab[ch] = len(vocab)
    for t in ["▁ev", "ev", "ler", "gü", "zel", "▁gü", "▁(", "da"]:
        vocab[t] = len(vocab)
    t = MorpheusTokenizer(vocab=vocab)
    t._morpheus_segment_batch = _stub_segment
    return t


ROUNDTRIP_CASES = [
    "evler, güzel.",
    "Evlerimizdekiler, kitapçısı muvaffakiyetsizleştiriciler.",
    "Ankara'da 3,5 °C; fiyat 12.000 ₺ (KDV dahil).",
    "https://example.com/a?b=c&d=e",
    "def f(x):\n    return x\n",
    "if a:\n\tb = 1\n\n\n  c = 2",
    "  baştaki ve sondaki boşluklar  ",
    "iki  boşluk   üç boşluk",
    "satır\r\nsonu\r\n",
    "\n\nbaşta newline",
    "\tbaşta tab",
    "boşluk sonra newline  \nsonraki",
    "Word WWW xyz QUIZ âlim îman",
    "emoji 🙂 ve ± ile i̇ birleşik nokta",
    "düz ▁ işareti ve ▁▁ çift",
    "no-break\u00a0space ve\u2009ince boşluk",
    "a" * 75 + " uzun kelime",
    "",
    " ",
    "\n",
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip_is_exact(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == text


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip_with_special_tokens(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_existing_ids_are_stable():
    vocab = dict(SPECIAL_TOKENS)
    vocab["a"] = 5
    t = MorpheusTokenizer(vocab=vocab)
    assert t.vocab["a"] == 5
    assert t.vocab[WORD_BOUNDARY] == 6
    assert len(t.vocab) == len(vocab) + len(set(t.vocab) - set(vocab))


def test_constructor_does_not_mutate_input_vocab():
    vocab = dict(SPECIAL_TOKENS)
    MorpheusTokenizer(vocab=vocab)
    assert vocab == SPECIAL_TOKENS


def test_oov_char_uses_byte_fallback(tok):
    ids = tok.encode("🙂", add_special_tokens=False)
    byte_ids = [tok.vocab[b] for b in (BYTE_TOKENS[x] for x in "🙂".encode("utf-8"))]
    assert ids[-len(byte_ids):] == byte_ids


def test_word_start_prefers_known_segment(tok):
    # "güzel" -> "▁gü" is in vocab directly; an unknown word start splits into ▁ + segment.
    assert tok.encode("güzel", add_special_tokens=False) == [tok.vocab["▁gü"], tok.vocab["zel"]]
    assert tok.encode("x daha", add_special_tokens=False)[2:4] == [tok.vocab[WORD_BOUNDARY], tok.vocab["da"]]


def test_indentation_is_preserved(tok):
    text = "a:\n    b"
    assert tok.decode(tok.encode(text, add_special_tokens=False)) == text


def test_export_matches_encode(tok, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("güzel evler 🙂\nikinci satır\n", encoding="utf-8")
    out = tmp_path / "ids.txt"
    stats = export_corpus_tokenized(tok, str(corpus), str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [list(map(int, l.split())) for l in lines] == [
        tok.encode("güzel evler 🙂", add_special_tokens=False),
        tok.encode("ikinci satır", add_special_tokens=False),
    ]
    assert stats["unk_count"] == 0
    assert stats["byte_fallback_count"] == 4


def test_case_insensitive_cache_keeps_surface_form(tok):
    tok.preserve_case = False
    tok.segment_words_batched(["Evler"])
    assert tok.segment_words_batched(["evler"]) == [["ev", "ler"]]
    assert tok.decode(tok.encode("evler Evler EVLER", add_special_tokens=False)) == "evler Evler EVLER"
