def turkish_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def turkish_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


def _gpt2_bytes_to_unicode() -> dict:
    # The byte -> printable-character table used by GPT-2 / HF ByteLevel tokenizers.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_LEVEL_DECODER = {c: b for b, c in _gpt2_bytes_to_unicode().items()}


def byte_level_to_text(token: str) -> str:
    """Surface text of a byte-level BPE token ("Ġkitap" -> " kitap", "Ã¼" -> "ü").

    A token that holds only part of a multi-byte character decodes to U+FFFD.
    """
    return bytes(_BYTE_LEVEL_DECODER[c] for c in token if c in _BYTE_LEVEL_DECODER).decode("utf-8", errors="replace")
