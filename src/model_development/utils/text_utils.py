def turkish_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def turkish_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()
