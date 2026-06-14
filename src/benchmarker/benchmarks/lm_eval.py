import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import math
import time
import json
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.common.providers.logger_provider import global_logger
from src.common.text_utils import turkish_lower


class TokenizerAdapter:
    name: str = "base"
    vocab_size: int = 0

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError


class SentencePieceAdapter(TokenizerAdapter):
    def __init__(self, name: str, model_path: str):
        import sentencepiece as spm
        self.name = name
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()
        global_logger.info(f"[{name}] Loaded SentencePiece (vocab={self.vocab_size:,})")

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)


class WordPieceAdapter(TokenizerAdapter):
    def __init__(self, name: str, model_path: str):
        from tokenizers import Tokenizer
        self.name = name
        self.tok = Tokenizer.from_file(model_path)
        self.vocab_size = self.tok.get_vocab_size()
        vocab = self.tok.get_vocab()
        specials = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
        self.special_ids = {vocab[s] for s in specials if s in vocab}
        global_logger.info(f"[{name}] Loaded WordPiece (vocab={self.vocab_size:,})")

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text).ids
        return [i for i in ids if i not in self.special_ids]


class MorfessorAdapter(TokenizerAdapter):
    def __init__(
            self,
            name: str,
            model_path: str,
            train_corpus_path: str,
            target_vocab: int = 50_000,
            cache_path: Optional[Path] = None,
    ):
        from src.model_development.training.dataset import MorfessorWrapper
        self.name = name
        self.wrapper = MorfessorWrapper(model_path)
        self._seg_cache: Dict[str, List[str]] = {}

        if cache_path and cache_path.exists():
            self._load_vocab(cache_path)
        else:
            self._build_vocab(train_corpus_path, target_vocab)
            if cache_path:
                self._save_vocab(cache_path)

    def _segment(self, word: str) -> List[str]:
        cached = self._seg_cache.get(word)
        if cached is not None:
            return cached
        segs, _ = self.wrapper.segment(word)
        self._seg_cache[word] = segs
        return segs

    def _build_vocab(self, corpus_path: str, target_vocab: int):
        global_logger.info(f"[{self.name}] Building int vocab from {corpus_path} (target={target_vocab:,})")
        counter: Counter = Counter()
        n_lines = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                for word in line.strip().split():
                    w = turkish_lower(word)
                    counter.update(self._segment(w))
                if n_lines % 50_000 == 0:
                    global_logger.info(f"[{self.name}] vocab build: scanned {n_lines:,} lines, segments={len(counter):,}")

        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        chars = sorted({c for s in counter.keys() for c in s})
        remaining = max(0, target_vocab - len(specials) - len(chars))
        common = [s for s, _ in counter.most_common(remaining)]

        seen = set()
        self.itos: List[str] = []
        for t in specials + chars + common:
            if t in seen:
                continue
            self.itos.append(t)
            seen.add(t)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.unk_id = self.stoi["<UNK>"]
        global_logger.info(
            f"[{self.name}] Vocab built (size={self.vocab_size:,}, "
            f"unique_segments={len(counter):,}, chars={len(chars)})"
        )

    def _save_vocab(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)
        global_logger.info(f"[{self.name}] Vocab cached -> {path}")

    def _load_vocab(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.itos = data["itos"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.unk_id = self.stoi["<UNK>"]
        global_logger.info(f"[{self.name}] Vocab loaded from cache (size={self.vocab_size:,})")

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in text.split():
            w = turkish_lower(word)
            for s in self._segment(w):
                tid = self.stoi.get(s)
                if tid is not None:
                    ids.append(tid)
                else:
                    for ch in s:
                        ids.append(self.stoi.get(ch, self.unk_id))
        return ids


class MorpheusAdapter(TokenizerAdapter):
    def __init__(
            self,
            name: str,
            checkpoint_path: str,
            tokenizer_dir: str,
            device: Optional[torch.device] = None,
    ):
        from src.model_development.model.morpheus import Morpheus
        from src.model_development.training.trainer import TrainingConfig
        from src.model_development.tokenization.morpheus_tokenizer import MorpheusTokenizer

        sys.modules["__main__"].TrainingConfig = TrainingConfig

        self.name = name
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        morpheus = Morpheus(
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
        morpheus.load_state_dict(ckpt["model_state"])
        morpheus.to(device).eval()
        self._morpheus_model = morpheus

        self.tokenizer = MorpheusTokenizer.load(
            tokenizer_dir,
            morpheus_model=morpheus,
            device=device,
        )
        self.vocab_size = self.tokenizer.vocab_size
        global_logger.info(f"[{name}] Loaded MorpheusTokenizer (vocab={self.vocab_size:,})")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class TurkishTokenizerLMAdapter(TokenizerAdapter):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.name = wrapper.name
        self.vocab_size = wrapper.vocab_size

    def encode(self, text: str) -> List[int]:
        return self.wrapper.encode_ids(text)


def encode_corpus_to_tensor(
        adapter: TokenizerAdapter,
        corpus_path: Path,
        cache_path: Optional[Path] = None,
        max_lines: Optional[int] = None,
) -> torch.Tensor:
    if cache_path and cache_path.exists():
        global_logger.info(f"[{adapter.name}] Loading token stream from cache: {cache_path}")
        return torch.load(cache_path)

    global_logger.info(f"[{adapter.name}] Encoding {corpus_path} (max_lines={max_lines})...")
    t0 = time.time()
    all_ids: List[int] = []
    n_lines = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            all_ids.extend(adapter.encode(line))
            if max_lines is not None and n_lines >= max_lines:
                break
            if n_lines % 25_000 == 0:
                elapsed = time.time() - t0
                global_logger.info(
                    f"[{adapter.name}] encoded {n_lines:,} lines, "
                    f"{len(all_ids):,} tokens, {elapsed:.0f}s elapsed"
                )

    tensor = torch.tensor(all_ids, dtype=torch.long)
    elapsed = time.time() - t0
    global_logger.info(
        f"[{adapter.name}] Encoded {n_lines:,} lines -> {len(all_ids):,} tokens "
        f"(avg {len(all_ids) / max(n_lines, 1):.1f}/line, {elapsed:.1f}s wall)"
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, cache_path)
        global_logger.info(f"[{adapter.name}] Cached token stream -> {cache_path}")

    return tensor


def count_corpus_chars(corpus_path: Path, max_lines: Optional[int] = None) -> int:
    n = 0
    n_lines = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            n += len(stripped)
            n_lines += 1
            if max_lines is not None and n_lines >= max_lines:
                break
    return n


class GPTBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_head, dropout=dropout, batch_first=True, bias=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        m = self.causal_mask[:S, :S]
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, attn_mask=m, is_causal=True, need_weights=False)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int = 512,
            n_layer: int = 8,
            n_head: int = 8,
            max_seq_len: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock(dim, n_head, dropout, max_seq_len) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
            self,
            ids: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = ids.shape
        pos = torch.arange(S, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction="mean",
            )
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TokenStreamDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def compute_bpc(
        model: MiniGPT,
        test_tokens: torch.Tensor,
        test_char_count: int,
        seq_len: int,
        device: torch.device,
        batch_size: int = 16,
) -> Tuple[float, float, float]:
    model.eval()
    n_chunks = max(0, (len(test_tokens) - 1) // seq_len)
    total_nll_nats = 0.0
    n_predicted = 0

    for batch_start in range(0, n_chunks, batch_size):
        batch_end = min(batch_start + batch_size, n_chunks)
        xs, ys = [], []
        for i in range(batch_start, batch_end):
            s = i * seq_len
            xs.append(test_tokens[s:s + seq_len])
            ys.append(test_tokens[s + 1:s + 1 + seq_len])
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        logits, _ = model(x)
        nll = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            y.reshape(-1),
            reduction="sum",
        )
        total_nll_nats += nll.item()
        n_predicted += y.numel()

    if n_predicted == 0:
        return float("nan"), float("nan"), float("nan")

    avg_nll = total_nll_nats / n_predicted
    bpc = (total_nll_nats * math.log2(math.e)) / max(test_char_count, 1)
    token_ppl = math.exp(avg_nll)
    return bpc, token_ppl, avg_nll


@dataclass
class LMTrainConfig:
    dim: int
    n_layer: int
    n_head: int
    seq_len: int
    batch_size: int
    n_epochs: float
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    weight_decay: float
    dropout: float
    eval_every_n_steps: int
    log_every_n_steps: int
    equalize_params: bool = True
    max_steps: Optional[int] = None


def _estimate_params(vocab_size: int, dim: int, n_layer: int, seq_len: int) -> int:
    return vocab_size * dim + 12 * n_layer * dim * dim + seq_len * dim


def compute_equalized_dim(
        vocab_size: int,
        target_params: int,
        n_layer: int,
        seq_len: int,
        n_head: int,
        min_dim: int = 128,
        max_dim: int = 1024,
) -> int:
    a = 12 * n_layer
    b = vocab_size + seq_len
    c = -target_params
    disc = b * b - 4 * a * c
    if disc < 0:
        return min_dim
    raw = (-b + math.sqrt(disc)) / (2 * a)
    rounded = max(n_head, int(round(raw / n_head) * n_head))
    return max(min_dim, min(max_dim, rounded))


PILOT_CONFIG = LMTrainConfig(
    dim=256, n_layer=4, n_head=4, seq_len=256,
    batch_size=64, n_epochs=0.5,
    learning_rate=3e-4, warmup_steps=200, grad_clip=1.0,
    weight_decay=0.01, dropout=0.1,
    eval_every_n_steps=300, log_every_n_steps=50,
)

FULL_CONFIG = LMTrainConfig(
    dim=512, n_layer=8, n_head=8, seq_len=512,
    batch_size=32, n_epochs=4.0, max_steps=10000,
    learning_rate=3e-4, warmup_steps=500, grad_clip=1.0,
    weight_decay=0.01, dropout=0.1,
    eval_every_n_steps=1000, log_every_n_steps=100,
)


def train_one_tokenizer(
        adapter: TokenizerAdapter,
        train_tokens: torch.Tensor,
        test_tokens: torch.Tensor,
        test_char_count: int,
        cfg: LMTrainConfig,
        output_dir: Path,
        device: torch.device,
        target_params: Optional[int] = None,
) -> Dict:
    name = adapter.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{name}_train_log.csv"
    ckpt_path = output_dir / f"{name}_model.pt"

    if target_params is not None:
        effective_dim = compute_equalized_dim(
            vocab_size=adapter.vocab_size,
            target_params=target_params,
            n_layer=cfg.n_layer,
            seq_len=cfg.seq_len,
            n_head=cfg.n_head,
        )
        global_logger.info(
            f"[{name}] Param-equalized dim: {effective_dim} (baseline cfg.dim={cfg.dim}, "
            f"target_params≈{target_params/1e6:.1f}M)"
        )
    else:
        effective_dim = cfg.dim

    global_logger.info(
        f"\n{'=' * 78}\n"
        f"[{name}] LM training start "
        f"(vocab={adapter.vocab_size:,}, dim={effective_dim}, "
        f"train_tokens={len(train_tokens):,}, test_tokens={len(test_tokens):,})\n"
        f"{'=' * 78}"
    )

    model = MiniGPT(
        vocab_size=adapter.vocab_size,
        dim=effective_dim, n_layer=cfg.n_layer, n_head=cfg.n_head,
        max_seq_len=cfg.seq_len, dropout=cfg.dropout,
    ).to(device)
    n_params = model.parameter_count()
    global_logger.info(f"[{name}] Model params: {n_params:,} ({n_params / 1e6:.1f}M)")

    dataset = TokenStreamDataset(train_tokens, seq_len=cfg.seq_len)
    if len(dataset) == 0:
        global_logger.error(f"[{name}] Empty dataset (train tokens < seq_len). Skipping.")
        return {
            "tokenizer": name, "vocab_size": adapter.vocab_size,
            "n_train_tokens": len(train_tokens), "n_test_tokens": len(test_tokens),
            "n_test_chars": test_char_count, "n_params": n_params,
            "final_bpc": float("nan"), "best_bpc": float("nan"),
            "final_token_ppl": float("nan"), "final_token_nll_nats": float("nan"),
            "wall_time_s": 0.0,
        }

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    micro_steps_per_epoch = len(loader)
    if cfg.max_steps:
        total_steps = cfg.max_steps
    else:
        total_steps = max(1, int(micro_steps_per_epoch * cfg.n_epochs))
    warmup = min(cfg.warmup_steps, max(1, total_steps // 5))
    global_logger.info(
        f"[{name}] {micro_steps_per_epoch:,} steps/epoch, "
        f"total_steps={total_steps:,}, warmup={warmup:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    rows: List[Dict] = []
    t_start = time.time()
    step = 0
    best_bpc = float("inf")
    last_bpc = float("nan")
    last_ppl = float("nan")
    last_nll = float("nan")

    while step < total_steps:
        for x, y in loader:
            if step >= total_steps:
                break
            model.train()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every_n_steps == 0:
                global_logger.info(
                    f"[{name}][step {step:>6,}/{total_steps:,}] "
                    f"loss={loss.item():.4f} grad={grad_norm.item():.2f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step > 0 and step % cfg.eval_every_n_steps == 0:
                bpc, ppl, nll = compute_bpc(
                    model, test_tokens, test_char_count,
                    cfg.seq_len, device,
                )
                last_bpc, last_ppl, last_nll = bpc, ppl, nll
                global_logger.info(
                    f"[{name}][eval @ step {step:,}] "
                    f"BPC={bpc:.4f}  token_ppl={ppl:.2f}  nll={nll:.4f}"
                )
                rows.append({
                    "step": step, "train_loss": loss.item(),
                    "val_bpc": bpc, "val_token_ppl": ppl, "val_token_nll": nll,
                    "lr": scheduler.get_last_lr()[0],
                })
                if bpc < best_bpc:
                    best_bpc = bpc

            step += 1

    final_bpc, final_ppl, final_nll = compute_bpc(
        model, test_tokens, test_char_count, cfg.seq_len, device,
    )
    if final_bpc < best_bpc:
        best_bpc = final_bpc

    elapsed = time.time() - t_start

    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": adapter.vocab_size,
        "model_config": {
            "vocab_size": adapter.vocab_size,
            "dim": effective_dim,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "max_seq_len": cfg.seq_len,
            "dropout": cfg.dropout,
        },
        "train_config": cfg.__dict__,
        "final_bpc": final_bpc,
        "best_bpc": best_bpc,
        "n_params": n_params,
        "elapsed_s": elapsed,
    }, ckpt_path)

    if rows:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    global_logger.info(
        f"\n[{name}] DONE  final_BPC={final_bpc:.4f}  best_BPC={best_bpc:.4f}  "
        f"token_ppl={final_ppl:.2f}  wall={elapsed / 60:.1f}min"
    )

    return {
        "tokenizer": name,
        "vocab_size": adapter.vocab_size,
        "n_train_tokens": len(train_tokens),
        "n_test_tokens": len(test_tokens),
        "n_test_chars": test_char_count,
        "n_params": n_params,
        "final_bpc": final_bpc,
        "best_bpc": best_bpc,
        "final_token_ppl": final_ppl,
        "final_token_nll_nats": final_nll,
        "wall_time_s": elapsed,
    }


def benchmark_encoding(
        adapter: TokenizerAdapter,
        test_text: str,
        n_warmup: int = 3,
        n_measure: int = 7,
) -> Dict:
    import statistics
    warmup_text = test_text[: min(2000, len(test_text))]
    for _ in range(n_warmup):
        adapter.encode(warmup_text)

    times: List[float] = []
    n_tokens_last = 0
    for _ in range(n_measure):
        t0 = time.perf_counter()
        ids = adapter.encode(test_text)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_tokens_last = len(ids)

    n_chars = len(test_text)
    median_s = statistics.median(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "chars_per_sec_encode": n_chars / median_s if median_s > 0 else float("inf"),
        "tokens_per_sec_encode": n_tokens_last / median_s if median_s > 0 else float("inf"),
        "ms_per_kchar_encode": (median_s / n_chars * 1e6) if n_chars > 0 else float("nan"),
        "fertility_tokens_per_char": n_tokens_last / n_chars,
        "encode_median_s": median_s,
        "encode_std_s": stdev_s,
        "encode_n_chars": n_chars,
        "encode_n_tokens": n_tokens_last,
    }


def load_trained_lm(ckpt_path: Path, device: torch.device) -> Tuple[MiniGPT, Dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state"]

    inferred_vocab, inferred_dim = state["tok_emb.weight"].shape
    inferred_max_seq = state["pos_emb.weight"].shape[0]
    block_layer_indices = set()
    for k in state.keys():
        if k.startswith("blocks."):
            block_layer_indices.add(int(k.split(".")[1]))
    inferred_n_layer = max(block_layer_indices) + 1 if block_layer_indices else 1

    mcfg = ckpt.get("model_config") or {}
    train_cfg = ckpt.get("train_config", ckpt.get("config", {}))
    n_head = mcfg.get("n_head") or train_cfg.get("n_head", 8)
    dropout = mcfg.get("dropout") or train_cfg.get("dropout", 0.1)

    model = MiniGPT(
        vocab_size=inferred_vocab,
        dim=inferred_dim,
        n_layer=inferred_n_layer,
        n_head=n_head,
        max_seq_len=inferred_max_seq,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    global_logger.info(
        f"[load_trained_lm] {ckpt_path.name}: vocab={inferred_vocab:,} dim={inferred_dim} "
        f"n_layer={inferred_n_layer} n_head={n_head} seq={inferred_max_seq}"
    )
    return model, ckpt


@torch.no_grad()
def benchmark_generation(
        model: MiniGPT,
        device: torch.device,
        n_generate_tokens: int = 200,
        batch_size: int = 1,
        n_warmup: int = 3,
) -> Dict:
    import statistics
    max_seq_len = model.max_seq_len

    prompt_len = 8
    prompt = torch.randint(0, model.vocab_size, (batch_size, prompt_len), device=device)

    for _ in range(n_warmup):
        x = prompt
        for _ in range(8):
            logits, _ = model(x)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            if x.size(1) > max_seq_len:
                x = x[:, -max_seq_len:]

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    per_run_times: List[float] = []
    n_runs = 3
    for _ in range(n_runs):
        x = prompt.clone()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_generate_tokens):
            logits, _ = model(x)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            if x.size(1) > max_seq_len:
                x = x[:, -max_seq_len:]
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_run_times.append(time.perf_counter() - t0)

    median_s = statistics.median(per_run_times)
    total_tokens_generated = n_generate_tokens * batch_size
    ms_per_token = (median_s / n_generate_tokens) * 1000
    tokens_per_sec = total_tokens_generated / median_s

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "ms_per_token_gen": ms_per_token,
        "tokens_per_sec_gen": tokens_per_sec,
        "gen_median_s": median_s,
        "gen_batch_size": batch_size,
        "gen_n_tokens": n_generate_tokens,
        "peak_gpu_memory_mb": peak_mem_mb,
    }


def get_tokenizer_artifact_size_mb(adapter: TokenizerAdapter, artifacts_dir: Path) -> float:
    name = adapter.name
    classical = artifacts_dir / "tokenizers" / "classical"
    morpheus_50k = artifacts_dir / "tokenizers" / "morpheus_50k"

    candidates: List[Path] = []
    if "morpheus" in name:
        if morpheus_50k.exists():
            candidates.extend(morpheus_50k.rglob("*"))
        ckpt_dir = artifacts_dir / "checkpoints"
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*v3_final*.pt"):
                candidates.append(f)
                break
            else:
                for f in ckpt_dir.glob("*_final.pt"):
                    candidates.append(f)
                    break
                else:
                    for f in ckpt_dir.glob("*_best.pt"):
                        candidates.append(f)
                        break
    elif "morfessor" in name:
        f = classical / "morfessor_model.bin"
        if f.exists():
            candidates.append(f)
    elif "bpe" in name or "unigram" in name or "wordpiece" in name or "byte" in name:
        prefix = name.split("-")[0].replace("-", "_")
        if name.startswith("byte-bpe"):
            prefix = "byte_bpe"
        for ext in [".model", ".vocab", ".json"]:
            for f in classical.glob(f"{prefix}_*{ext}"):
                candidates.append(f)

    total_bytes = sum(f.stat().st_size for f in candidates if f.is_file())
    return total_bytes / (1024 * 1024)


def _vocab_size_from_name(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def build_all_adapters(
        artifacts_dir: Path,
        train_corpus_path: Path,
        cache_dir: Path,
        classical_vocab: Optional[int] = None,
) -> List[TokenizerAdapter]:
    adapters: List[TokenizerAdapter] = []
    classical = artifacts_dir / "tokenizers" / "classical"
    morpheus_50k = artifacts_dir / "tokenizers" / "morpheus_50k"
    checkpoints = artifacts_dir / "checkpoints"

    for prefix, kind, ext in [
        ("bpe", "spm", "*.model"),
        ("byte_bpe", "spm", "*.model"),
        ("unigram", "spm", "*.model"),
        ("wordpiece", "wp", "*.json"),
    ]:
        cands = sorted(classical.glob(f"{prefix}_{ext}"))
        cands = [c for c in cands if _vocab_size_from_name(c) > 0]
        if not cands:
            global_logger.warning(f"[lm_eval] no {prefix} model found in {classical}")
            continue
        if classical_vocab is not None:
            chosen = min(cands, key=lambda c: abs(_vocab_size_from_name(c) - classical_vocab))
        else:
            chosen = max(cands, key=_vocab_size_from_name)
        vsize = _vocab_size_from_name(chosen)
        name = f"{prefix.replace('_', '-')}-{vsize // 1000}k"
        if kind == "spm":
            adapters.append(SentencePieceAdapter(name, str(chosen)))
        else:
            adapters.append(WordPieceAdapter(name, str(chosen)))

    morf_path = classical / "morfessor_model.bin"
    if morf_path.exists():
        adapters.append(MorfessorAdapter(
            "morfessor-50k",
            str(morf_path),
            str(train_corpus_path),
            target_vocab=50_000,
            cache_path=cache_dir / "morfessor_50k_vocab.json",
        ))
    else:
        global_logger.warning(f"[lm_eval] morfessor_model.bin not found at {morf_path}")

    ckpt_path: Optional[Path] = None
    preferred = [
        "turkish_morpheus_a100_v3_final.pt",
        "turkish_morpheus_a100_v3_best.pt",
        "turkish_morpheus_a100_release_best.pt",
        "turkish_morpheus_a100_best.pt",
    ]
    for cand in preferred:
        p = checkpoints / cand
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None and checkpoints.exists():
        best_pts = sorted(checkpoints.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_pts:
            ckpt_path = best_pts[0]
            global_logger.info(f"[lm_eval] Morpheus checkpoint auto-discovered: {ckpt_path.name}")
        else:
            any_pts = sorted(checkpoints.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if any_pts:
                ckpt_path = any_pts[0]
                global_logger.info(f"[lm_eval] Morpheus checkpoint fallback (most recent .pt): {ckpt_path.name}")

    from src.benchmarker.metrics.external_tokenizers import load_turkish_tokenizer_or_none
    tt = load_turkish_tokenizer_or_none()
    if tt is not None:
        adapters.append(TurkishTokenizerLMAdapter(tt))

    if ckpt_path and morpheus_50k.exists():
        adapters.append(MorpheusAdapter(
            "morpheus-50k",
            str(ckpt_path),
            str(morpheus_50k),
        ))
    else:
        listed = sorted(p.name for p in checkpoints.glob("*.pt")) if checkpoints.exists() else []
        global_logger.warning(
            f"[lm_eval] Morpheus tokenizer missing "
            f"(ckpt={ckpt_path}, dir_exists={morpheus_50k.exists()}, "
            f"available_checkpoints={listed})"
        )

    return adapters


def run_inference_bench(
        adapters: List[TokenizerAdapter],
        trained_models_dir: Path,
        output_dir: Path,
        artifacts_dir: Path,
        test_corpus_path: Path,
        device: torch.device,
        encode_chars: int = 200_000,
        n_generate_tokens: int = 200,
        batch_sizes: Optional[List[int]] = None,
) -> List[Dict]:
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()
    test_text = raw[:encode_chars] if len(raw) > encode_chars else raw
    global_logger.info(f"[inference-bench] test text: {len(test_text):,} chars")

    rows: List[Dict] = []
    for adapter in adapters:
        global_logger.info(f"\n[inference-bench] === {adapter.name} ===")

        enc = benchmark_encoding(adapter, test_text)
        global_logger.info(
            f"[{adapter.name}] encode: {enc['chars_per_sec_encode']:,.0f} chars/s, "
            f"{enc['tokens_per_sec_encode']:,.0f} tok/s, fertility={enc['fertility_tokens_per_char']:.4f}"
        )

        artifact_mb = get_tokenizer_artifact_size_mb(adapter, artifacts_dir)
        global_logger.info(f"[{adapter.name}] tokenizer artifact size: {artifact_mb:.2f} MB")

        ckpt_path = trained_models_dir / f"{adapter.name}_model.pt"
        if not ckpt_path.exists():
            global_logger.warning(
                f"[{adapter.name}] No trained LM checkpoint at {ckpt_path}, "
                f"skipping generation bench."
            )
            row = {"tokenizer": adapter.name, **enc, "tokenizer_artifact_mb": artifact_mb}
            rows.append(row)
            continue

        model, ckpt = load_trained_lm(ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        bpc = ckpt.get("best_bpc", ckpt.get("final_bpc", float("nan")))

        gen_results = {}
        for bs in batch_sizes:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                g = benchmark_generation(
                    model, device,
                    n_generate_tokens=n_generate_tokens,
                    batch_size=bs,
                )
                fertility = enc["fertility_tokens_per_char"]
                chars_per_sec_gen = g["tokens_per_sec_gen"] / fertility if fertility > 0 else float("nan")
                ms_per_char_gen = g["ms_per_token_gen"] * fertility
                gen_results[bs] = {
                    **g,
                    "chars_per_sec_gen_eff": chars_per_sec_gen,
                    "ms_per_char_gen_eff": ms_per_char_gen,
                }
                global_logger.info(
                    f"[{adapter.name}] gen B={bs}: {g['ms_per_token_gen']:.2f} ms/tok, "
                    f"{g['tokens_per_sec_gen']:.0f} tok/s, "
                    f"effective={chars_per_sec_gen:.0f} char/s "
                    f"(GPU peak {g['peak_gpu_memory_mb']:.0f} MB)"
                )
            except Exception as e:
                global_logger.error(f"[{adapter.name}] gen B={bs} FAILED: {e}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row = {
            "tokenizer": adapter.name,
            "vocab_size": adapter.vocab_size,
            "lm_params_M": n_params / 1e6,
            "best_bpc": bpc,
            "tokenizer_artifact_mb": artifact_mb,
            **enc,
        }
        for bs, g in gen_results.items():
            row[f"ms_per_token_gen_B{bs}"] = g["ms_per_token_gen"]
            row[f"tokens_per_sec_gen_B{bs}"] = g["tokens_per_sec_gen"]
            row[f"chars_per_sec_gen_eff_B{bs}"] = g["chars_per_sec_gen_eff"]
            row[f"ms_per_char_gen_eff_B{bs}"] = g["ms_per_char_gen_eff"]
            row[f"peak_gpu_memory_mb_B{bs}"] = g["peak_gpu_memory_mb"]
        rows.append(row)

    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        ordered = ["tokenizer", "vocab_size", "lm_params_M", "best_bpc",
                   "tokenizer_artifact_mb", "fertility_tokens_per_char",
                   "chars_per_sec_encode", "tokens_per_sec_encode"]
        rest = [k for k in all_keys if k not in ordered]
        fieldnames = ordered + rest

        summary_path = output_dir / "inference_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        global_logger.info("\n" + "=" * 78)
        global_logger.info("INFERENCE BENCHMARK SUMMARY")
        global_logger.info("=" * 78)
        global_logger.info(
            f"  {'tokenizer':<20s} {'BPC':>7s} {'enc kchar/s':>12s} "
            f"{'gen ms/tok':>11s} {'gen char/s eff':>15s} {'fert':>6s}"
        )
        for r in sorted(rows, key=lambda r: r.get("best_bpc", float("inf"))):
            bpc_val = r.get("best_bpc", float("nan"))
            bpc_str = f"{bpc_val:.4f}" if bpc_val == bpc_val else "n/a"
            enc_kchar = r.get("chars_per_sec_encode", 0) / 1000
            gen_ms = r.get("ms_per_token_gen_B1", float("nan"))
            gen_cs = r.get("chars_per_sec_gen_eff_B1", float("nan"))
            global_logger.info(
                f"  {r['tokenizer']:<20s} {bpc_str:>7s} {enc_kchar:>12.1f} "
                f"{gen_ms:>11.2f} {gen_cs:>15.0f} {r['fertility_tokens_per_char']:>6.3f}"
            )
        global_logger.info("=" * 78)
        global_logger.info(f"Detailed CSV: {summary_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.benchmarks.lm_eval")
    parser.add_argument("--mode", choices=["pilot", "full", "inference"], default="pilot")
    parser.add_argument("--trained-mode", choices=["pilot", "full"], default="full",
                        help="Which trained-model directory to load from for inference mode")
    parser.add_argument("--encode-chars", type=int, default=200_000)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Train every tokenizer for exactly this many optimizer steps "
             "(equal compute budget + identical LR schedule across tokenizers). "
             "Overrides epoch-based step count.",
    )
    parser.add_argument(
        "--train-cap-lines", type=int, default=1_000_000,
        help="Encode at most this many corpus lines for LM training "
             "(bounds Morpheus encode time; same for all tokenizers).",
    )
    parser.add_argument(
        "--test-cap-lines", type=int, default=100_000,
        help="Encode at most this many lines of the test split for BPC.",
    )
    parser.add_argument(
        "--classical-vocab", type=int, default=None,
        help="Pick classical tokenizers closest to this vocab size instead of the largest "
             "(vocab-matched comparison); outputs go to a _cv<N>k-suffixed directory",
    )
    parser.add_argument("--gen-tokens", type=int, default=200)
    parser.add_argument(
        "--tokenizers", default=None,
        help="Comma-separated subset of tokenizer names to run (default: all)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Do not load/save token-stream caches",
    )
    parser.add_argument(
        "--skip-encoded", action="store_true",
        help="Skip tokenizers whose checkpoint already exists in output dir",
    )
    args = parser.parse_args()

    BASE = Path(__file__).resolve().parents[3]
    artifacts = BASE / "src" / "model_development" / "artifacts"
    train_corpus = artifacts / "datasets" / "splits" / "train.txt"
    test_corpus = artifacts / "datasets" / "splits" / "test.txt"

    dir_suffix = f"_cv{args.classical_vocab // 1000}k" if args.classical_vocab else ""
    output_dir = BASE / "src" / "benchmarker" / "results" / "lm_eval" / (args.mode + dir_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifacts / "lm_eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        global_logger.info("[lm_eval] TF32 enabled for matmul + cuDNN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "inference":
        cfg = FULL_CONFIG if args.trained_mode == "full" else PILOT_CONFIG
    else:
        cfg = PILOT_CONFIG if args.mode == "pilot" else FULL_CONFIG
    if args.max_steps:
        cfg.max_steps = args.max_steps
    global_logger.info(f"[lm_eval] mode={args.mode}  device={device}")
    global_logger.info(f"[lm_eval] config={cfg}")

    if not train_corpus.exists() or not test_corpus.exists():
        global_logger.error(
            f"[lm_eval] train/test corpus not found: "
            f"{train_corpus.exists()=}, {test_corpus.exists()=}. "
            f"Run `python -m src.run_pipeline --stage data` first."
        )
        sys.exit(1)

    train_char_count = count_corpus_chars(train_corpus, max_lines=args.train_cap_lines)
    test_char_count = count_corpus_chars(test_corpus, max_lines=args.test_cap_lines)
    global_logger.info(
        f"[lm_eval] train_chars={train_char_count:,} (cap={args.train_cap_lines:,} lines)  "
        f"test_chars={test_char_count:,} (cap={args.test_cap_lines:,} lines)"
    )

    adapters = build_all_adapters(
        artifacts, train_corpus, cache_dir,
        classical_vocab=args.classical_vocab,
    )
    if args.tokenizers:
        wanted = set(s.strip() for s in args.tokenizers.split(","))
        adapters = [a for a in adapters if a.name in wanted]

    global_logger.info(f"[lm_eval] adapters to run: {[a.name for a in adapters]}")

    if args.mode == "inference":
        trained_dir = BASE / "src" / "benchmarker" / "results" / "lm_eval" / (args.trained_mode + dir_suffix)
        if not trained_dir.exists():
            global_logger.error(
                f"[lm_eval] No trained models found at {trained_dir}. "
                f"Run `--mode {args.trained_mode}` first to train LMs."
            )
            sys.exit(1)
        run_inference_bench(
            adapters=adapters,
            trained_models_dir=trained_dir,
            output_dir=output_dir,
            artifacts_dir=artifacts,
            test_corpus_path=test_corpus,
            device=device,
            encode_chars=args.encode_chars,
            n_generate_tokens=args.gen_tokens,
        )
        return

    target_params: Optional[int] = None
    if cfg.equalize_params and adapters:
        max_vocab_adapter = max(adapters, key=lambda a: a.vocab_size)
        target_params = _estimate_params(
            vocab_size=max_vocab_adapter.vocab_size,
            dim=cfg.dim,
            n_layer=cfg.n_layer,
            seq_len=cfg.seq_len,
        )
        global_logger.info(
            f"[lm_eval] Param equalization ON. "
            f"Baseline = {max_vocab_adapter.name} (vocab={max_vocab_adapter.vocab_size:,}, "
            f"dim={cfg.dim}) -> target_params={target_params:,} ({target_params/1e6:.1f}M). "
            f"Other tokenizers will have dim adjusted to match."
        )

    summary_rows: List[Dict] = []
    for adapter in adapters:
        ckpt_path = output_dir / f"{adapter.name}_model.pt"
        if args.skip_encoded and ckpt_path.exists():
            global_logger.info(f"[{adapter.name}] checkpoint exists, skipping")
            continue

        try:
            cap_tag = f"_t{args.train_cap_lines // 1000}k"
            train_cache = None if args.no_cache else cache_dir / f"{adapter.name}_train_tokens{cap_tag}.pt"
            test_cache = None if args.no_cache else cache_dir / f"{adapter.name}_test_tokens{cap_tag}.pt"

            train_tokens = encode_corpus_to_tensor(
                adapter, train_corpus, train_cache, max_lines=args.train_cap_lines)
            test_tokens = encode_corpus_to_tensor(
                adapter, test_corpus, test_cache, max_lines=args.test_cap_lines)

            if len(train_tokens) and len(test_tokens):
                observed_vocab = int(max(train_tokens.max().item(), test_tokens.max().item())) + 1
                if observed_vocab > adapter.vocab_size:
                    global_logger.warning(
                        f"[{adapter.name}] observed max token id implies vocab "
                        f"{observed_vocab:,} > declared {adapter.vocab_size:,}; bumping."
                    )
                    adapter.vocab_size = observed_vocab

            row = train_one_tokenizer(
                adapter, train_tokens, test_tokens, test_char_count,
                cfg, output_dir, device,
                target_params=target_params,
            )
            summary_rows.append(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            global_logger.error(f"[lm_eval] {adapter.name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    if summary_rows:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        global_logger.info("\n" + "=" * 78)
        global_logger.info("LM EVAL FINAL SUMMARY  (sorted by best BPC — lower is better)")
        global_logger.info("=" * 78)
        sorted_rows = sorted(summary_rows, key=lambda r: r["best_bpc"])
        for r in sorted_rows:
            global_logger.info(
                f"  {r['tokenizer']:<22s}  "
                f"BPC={r['best_bpc']:.4f}  "
                f"tok_ppl={r['final_token_ppl']:>7.2f}  "
                f"vocab={r['vocab_size']:>6,}  "
                f"params={r['n_params'] / 1e6:.1f}M  "
                f"time={r['wall_time_s'] / 60:.1f}min"
            )
        global_logger.info("=" * 78)
        global_logger.info(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
