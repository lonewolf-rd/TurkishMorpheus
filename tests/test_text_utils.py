from src.common.text_utils import byte_level_to_text, turkish_lower


def test_byte_level_to_text_decodes_gpt2_byte_mapping():
    assert byte_level_to_text("Ġkitap") == " kitap"
    assert byte_level_to_text("Ã¼") == "ü"
    assert byte_level_to_text("ÄŁ") == "ğ"


def test_byte_level_to_text_marks_partial_characters():
    # First byte of "ü" (0xC3) alone is not valid UTF-8.
    assert byte_level_to_text("Ã") == "�"


def test_turkish_lower_keeps_length():
    assert turkish_lower("İSTANBUL Irak") == "istanbul ırak"
