from typing import List, Optional

from src.common.providers.logger_provider import global_logger


TURKISH_TOKENIZER_VOCAB = 32768


class TurkishTokenizerWrapper:
    name = "turkish-tokenizer"

    def __init__(self):
        from turkish_tokenizer import TurkishTokenizer
        self.tt = TurkishTokenizer()
        self.vocab_size = TURKISH_TOKENIZER_VOCAB

    def _piece(self, token_id: int) -> str:
        return self.tt.decode([token_id]).strip()

    def encode_ids(self, text: str) -> List[int]:
        return list(self.tt.encode(text))

    def pieces(self, text: str) -> List[str]:
        return [p for p in (self._piece(i) for i in self.tt.encode(text)) if p]

    def segment(self, word: str) -> List[str]:
        return [p for p in (self._piece(i) for i in self.tt.encode(word)) if p]


def load_turkish_tokenizer_or_none() -> Optional[TurkishTokenizerWrapper]:
    try:
        wrapper = TurkishTokenizerWrapper()
        global_logger.info(
            f"[external_tokenizers] TurkishTokenizer loaded (vocab={wrapper.vocab_size:,})"
        )
        return wrapper
    except Exception as e:
        global_logger.warning(
            f"[external_tokenizers] TurkishTokenizer unavailable: {e} "
            f"(pip install turkish-tokenizer)"
        )
        return None
