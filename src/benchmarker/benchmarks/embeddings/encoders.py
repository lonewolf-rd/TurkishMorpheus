import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.common.providers.logger_provider import global_logger
from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.training.trainer import TrainingConfig

sys.modules["__main__"].TrainingConfig = TrainingConfig


BERTURK_MODEL = "dbmdz/bert-base-turkish-cased"
BGE_MODEL = "BAAI/bge-m3"


class WordEncoder:
    name: str = "base"
    dim: int = 0

    def encode_words(self, words: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_tokens_in_context(self, sentence_tokens: List[str]) -> np.ndarray:
        raise NotImplementedError


class MorpheusEncoder(WordEncoder):
    name = "morpheus"

    def __init__(
            self,
            checkpoint_path: str,
            device: Optional[torch.device] = None,
            batch_size: int = 256,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.helper = CharEncoderHelper()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.max_word_len = cfg.max_word_len
        self.model = Morpheus(
            char_dim=cfg.char_dim,
            char_embed_dim=cfg.char_embed_dim,
            case_embed_dim=cfg.case_embed_dim,
            n_layers_encoder=cfg.n_layers_encoder,
            n_layers_detector=cfg.n_layers_detector,
            num_heads=cfg.num_heads,
            max_word_len=cfg.max_word_len,
            max_segs=cfg.max_segs,
            dropout=cfg.dropout,
            threshold=cfg.threshold,
            pos_weight=cfg.pos_weight,
            count_loss_w=getattr(cfg, "count_loss_w", 0.3),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.dim = cfg.char_dim
        global_logger.info(
            f"[MorpheusEncoder] Loaded checkpoint {Path(checkpoint_path).name} (dim={self.dim})"
        )

    @torch.no_grad()
    def _encode_batch(self, words: List[str]) -> np.ndarray:
        ids_list, flags_list, real_lens = [], [], []
        for w in words:
            ids, flags, rl = self.helper.word_to_char_ids(w, max_len=self.max_word_len)
            ids_list.append(ids)
            flags_list.append(flags)
            real_lens.append(rl)

        char_ids = torch.tensor(ids_list, device=self.device)
        case_flags = torch.tensor(flags_list, device=self.device)
        real_length_t = torch.tensor(real_lens, device=self.device)

        out = self.model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_length_t,
        )
        return out["word_embeddings"].float().cpu().numpy()

    def encode_words(self, words: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for start in range(0, len(words), self.batch_size):
            chunk = [w if w else "·" for w in words[start: start + self.batch_size]]
            vecs.append(self._encode_batch(chunk))
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    def encode_tokens_in_context(self, sentence_tokens: List[str]) -> np.ndarray:
        return self.encode_words(sentence_tokens)


class HFEncoder(WordEncoder):
    def __init__(
            self,
            model_name: str,
            name: str,
            pooling: str = "mean",
            normalize: bool = False,
            device: Optional[torch.device] = None,
            batch_size: int = 64,
            max_length: int = 16,
    ):
        from transformers import AutoTokenizer, AutoModel

        self.name = name
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        global_logger.info(
            f"[HFEncoder] Loaded {model_name} as '{name}' "
            f"(dim={self.dim}, pooling={pooling}, normalize={normalize})"
        )

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def encode_words(self, words: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for start in range(0, len(words), self.batch_size):
            chunk = words[start: start + self.batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)
            out = self.model(**enc)
            pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
            vecs.append(pooled.float().cpu().numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    @torch.no_grad()
    def encode_tokens_in_context(self, sentence_tokens: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            sentence_tokens, is_split_into_words=True, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(self.device)
        out = self.model(**enc)
        hidden = out.last_hidden_state[0]
        word_ids = enc.word_ids(batch_index=0)

        vecs = np.zeros((len(sentence_tokens), self.dim), dtype=np.float32)
        counts = np.zeros(len(sentence_tokens), dtype=np.int64)
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            vecs[wid] += hidden[pos].float().cpu().numpy()
            counts[wid] += 1
        counts = np.clip(counts, 1, None)
        return vecs / counts[:, None]


def BERTurkEncoder(device: Optional[torch.device] = None, **kw) -> HFEncoder:
    return HFEncoder(BERTURK_MODEL, name="berturk", pooling="mean",
                     normalize=False, device=device, **kw)


def BGEEncoder(device: Optional[torch.device] = None, **kw) -> HFEncoder:
    return HFEncoder(BGE_MODEL, name="bge-m3", pooling="cls",
                     normalize=True, device=device, max_length=24, **kw)


def build_encoders(
        checkpoint_path: str,
        include_berturk: bool = True,
        include_bge: bool = True,
        device: Optional[torch.device] = None,
) -> List[WordEncoder]:
    encoders: List[WordEncoder] = [MorpheusEncoder(checkpoint_path, device=device)]
    if include_berturk:
        try:
            encoders.append(BERTurkEncoder(device=device))
        except Exception as e:
            global_logger.warning(f"[build_encoders] BERTurk unavailable, skipping: {e}")
    if include_bge:
        try:
            encoders.append(BGEEncoder(device=device))
        except Exception as e:
            global_logger.warning(f"[build_encoders] BGE-M3 unavailable, skipping: {e}")
    return encoders


def find_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    preferred = [
        "turkish_morpheus_a100_v3_final.pt",
        "turkish_morpheus_a100_v3_best.pt",
        "turkish_morpheus_a100_release_best.pt",
        "turkish_morpheus_a100_best.pt",
    ]
    for name in preferred:
        cand = checkpoints_dir / name
        if cand.exists():
            return cand
    pts = sorted(checkpoints_dir.glob("*.pt"))
    return pts[0] if pts else None
