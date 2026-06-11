import sys
import json
import re
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Callable, Any

from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.training.dataset import (
    MorfessorWrapper,
    clean_word_preserve_case,
)
from src.model_development.training.trainer import TrainingConfig
from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger
from src.benchmarker.metrics.intrinsic import (
    embed_word_list,
    root_cluster_coherence,
    morphological_analogy_accuracy,
    build_analogy_pairs,
)

sys.modules["__main__"].TrainingConfig = TrainingConfig


class ClassicalTokenizerWrapper:
    def __init__(self, name: str, kind: str, model_path: str):
        self.name = name
        self.kind = kind
        self.model_path = model_path
        self.model: Any = None
        self.vocab_set: set = set()
        self._load()

    def _load(self):
        if self.kind in ("bpe", "byte_bpe", "unigram"):
            import sentencepiece as spm
            self.model = spm.SentencePieceProcessor()
            self.model.load(self.model_path)
            self.vocab_set = {
                self.model.id_to_piece(i) for i in range(self.model.get_piece_size())
            }
        elif self.kind == "wordpiece":
            from tokenizers import Tokenizer
            self.model = Tokenizer.from_file(self.model_path)
            self.vocab_set = set(self.model.get_vocab().keys())
        else:
            raise ValueError(f"Unknown tokenizer kind: {self.kind}")

    def segment(self, word: str) -> List[str]:
        w = turkish_lower(word)
        if self.kind in ("bpe", "byte_bpe", "unigram"):
            pieces = self.model.encode_as_pieces(w)
            return [self._clean_piece(p) for p in pieces]
        else:
            tokens = self.model.encode(w).tokens
            return [self._clean_piece(t) for t in tokens]

    @staticmethod
    def _clean_piece(p: str) -> str:
        return p.replace("▁", "").replace("##", "").replace(" ", "")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_set)


def discover_classical_tokenizers(results_dir: str, preferred_vocab: int = 64000) -> List[ClassicalTokenizerWrapper]:
    results = Path(results_dir)
    if not results.exists():
        global_logger.warning(f"[ClassicalTokenizers] No results dir: {results_dir}")
        return []

    found: List[ClassicalTokenizerWrapper] = []
    patterns = [
        ("bpe", "bpe", "*.model"),
        ("byte_bpe", "byte_bpe", "*.model"),
        ("unigram", "unigram", "*.model"),
        ("wordpiece", "wordpiece", "*.json"),
    ]

    for kind_name, prefix, ext in patterns:
        candidates = sorted(results.glob(f"{prefix}_{ext}"))
        if not candidates:
            continue

        sized = []
        for path in candidates:
            m = re.search(r"_(\d+)", path.stem)
            if m:
                sized.append((int(m.group(1)), path))
        if not sized:
            continue

        sized.sort(key=lambda x: abs(x[0] - preferred_vocab))
        vocab_size, chosen = sized[0]
        name = f"{kind_name.upper()}-{vocab_size // 1000}K"
        try:
            wrapper = ClassicalTokenizerWrapper(name, kind_name, str(chosen))
            found.append(wrapper)
            global_logger.info(
                f"[ClassicalTokenizers] Loaded {name} from {chosen.name} "
                f"(vocab={wrapper.vocab_size})"
            )
        except Exception as e:
            global_logger.error(f"[ClassicalTokenizers] Failed to load {chosen}: {e}")

    return found


CURATED_SEEN_QUERIES = [
    "ev", "kitap", "su", "ay", "göz", "baş", "el", "gün",
    "yıl", "iş",
]

CURATED_SUFFIX_QUERIES = [
    "ev", "evler", "evde", "evden", "eve", "evim", "evimiz",
    "evimde", "evlerimizde", "evlerimizdeki", "evlerimizdekiler",
    "kitap", "kitaplar", "kitapta", "kitabım", "kitaplarımı",
    "kitabımdaki", "kitaplıkta",
    "gel", "geldi", "geliyor", "gelmiş", "gelecek", "gelirim",
    "gelmedik", "geliyorduk", "gelebilseydik",
    "git", "gitti", "gidiyor", "gitmiştik", "gidebilir",
    "gideceğim", "gitmedikçe",
]

CURATED_OOV_PROBES = [
    "muvaffakiyetsizleştiriciler",
    "anlaşılamamaktadır",
    "görevlendirilemeyeceklerinden",
    "üniversitelerindeki",
    "çalışmalarımızdan",
    "yapamayacağız",
    "söyleyebilseydik",
    "düşünmemiştik",
    "uçabilenlerden",
    "kitapçısı",
]

CURATED_NONCE = [
    "gloplaştırıcılık",
    "miyantlamak",
    "trafanlamış",
    "kelfeyimsi",
    "şarpınsızca",
]


class PaperEvaluator:
    def __init__(
            self,
            checkpoint_path: str,
            tokenizer_dir: Optional[str],
            morfessor_path: str,
            train_corpus_path: str,
            test_corpus_path: str,
            word_vocab_path: str,
            benchmarker_results_dir: Optional[str] = None,
            preferred_classical_vocab: int = 64000,
            output_dir: str = "paper_eval",
            max_word_len: int = 32,
            seed: int = 1337,
            device: Optional[torch.device] = None,
    ):
        torch.manual_seed(seed)
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = checkpoint_path
        self.tokenizer_dir = tokenizer_dir
        self.morfessor_path = morfessor_path
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path
        self.word_vocab_path = word_vocab_path
        self.benchmarker_results_dir = benchmarker_results_dir
        self.preferred_classical_vocab = preferred_classical_vocab
        self.max_word_len = max_word_len

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.helper = CharEncoderHelper()
        self.morfessor = MorfessorWrapper(morfessor_path)

        self.train_vocab: Dict[str, int] = {}
        self.test_words: List[str] = []
        self.stratified: Dict[str, List[str]] = {}
        self.word_to_root: Dict[str, str] = {}
        self.word_to_segments: Dict[str, List[str]] = {}
        self.word_to_morpheme_count: Dict[str, int] = {}

        self.trained_model: Optional[Morpheus] = None
        self.random_model: Optional[Morpheus] = None
        self.config = None
        self.classical_tokenizers: List[ClassicalTokenizerWrapper] = []

    def load_models(self):
        global_logger.info(f"[PaperEval] Loading trained model from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.config = cfg

        self.trained_model = Morpheus(
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
        self.trained_model.load_state_dict(ckpt["model_state"])
        self.trained_model.to(self.device).eval()

        self.random_model = Morpheus(
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
        self.random_model.to(self.device).eval()
        global_logger.info("[PaperEval] Models loaded.")

    def load_train_vocab(self):
        global_logger.info(f"[PaperEval] Loading train word vocab: {self.word_vocab_path}")
        data = torch.load(self.word_vocab_path)
        self.train_vocab = data["vocab"]
        global_logger.info(f"[PaperEval] Train vocab size: {len(self.train_vocab):,}")

    def build_stratified_test_set(
            self,
            n_seen: int = 800,
            n_oov: int = 400,
            min_test_freq: int = 2,
    ):
        global_logger.info(f"[PaperEval] Reading test corpus: {self.test_corpus_path}")
        counter: Counter = Counter()
        with open(self.test_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    cw = clean_word_preserve_case(w)
                    if cw:
                        counter[turkish_lower(cw)] += 1

        items = [(w, c) for w, c in counter.items() if c >= min_test_freq]
        items.sort(key=lambda x: -x[1])

        seen_words: List[str] = []
        oov_words: List[str] = []
        for w, c in items:
            if w in self.train_vocab:
                if len(seen_words) < n_seen:
                    seen_words.append(w)
            else:
                if len(oov_words) < n_oov:
                    oov_words.append(w)
            if len(seen_words) >= n_seen and len(oov_words) >= n_oov:
                break

        self.stratified = {
            "seen": seen_words,
            "oov": oov_words,
            "curated_seen": list(CURATED_SEEN_QUERIES),
            "curated_suffix": list(CURATED_SUFFIX_QUERIES),
            "curated_oov": list(CURATED_OOV_PROBES),
            "nonce": list(CURATED_NONCE),
        }

        all_words = (
            seen_words
            + oov_words
            + CURATED_SEEN_QUERIES
            + CURATED_SUFFIX_QUERIES
            + CURATED_OOV_PROBES
            + CURATED_NONCE
        )
        unique = list(dict.fromkeys(all_words))
        self.test_words = unique

        global_logger.info(
            f"[PaperEval] Stratified test set: "
            f"seen={len(seen_words)}, oov={len(oov_words)}, "
            f"curated_seen={len(CURATED_SEEN_QUERIES)}, "
            f"curated_suffix={len(CURATED_SUFFIX_QUERIES)}, "
            f"curated_oov={len(CURATED_OOV_PROBES)}, "
            f"nonce={len(CURATED_NONCE)}, "
            f"unique_total={len(unique)}"
        )

        global_logger.info("[PaperEval] Computing Morfessor segmentations for all test words...")
        for w in unique:
            segs, _ = self.morfessor.segment(w)
            self.word_to_segments[w] = segs
            self.word_to_root[w] = segs[0] if segs else w
            self.word_to_morpheme_count[w] = len(segs)

    @torch.no_grad()
    def compute_embeddings(self, model: Morpheus, words: List[str]) -> torch.Tensor:
        global_logger.info(f"[PaperEval] Embedding {len(words)} words with model...")
        model.eval()
        return embed_word_list(
            model=model,
            helper=self.helper,
            words=words,
            device=self.device,
        )

    @torch.no_grad()
    def model_segments(
            self,
            model: Morpheus,
            word: str,
            threshold: float = 0.5,
    ) -> List[str]:
        ids, flags, rl = self.helper.word_to_char_ids(word, max_len=self.max_word_len)
        char_ids = torch.tensor([ids], device=self.device)
        case_flags = torch.tensor([flags], device=self.device)
        real_lengths = torch.tensor([rl], device=self.device)

        out = model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_lengths,
        )
        boundary_probs = out["boundary_probs"][0].cpu().tolist()

        n_chars = len(word)
        segments: List[str] = []
        current = word[0] if word else ""
        for i in range(1, n_chars):
            if i < len(boundary_probs) and boundary_probs[i] > threshold:
                segments.append(current)
                current = word[i]
            else:
                current += word[i]
        if current:
            segments.append(current)
        return segments

    def knn_for_query(
            self,
            embeddings: torch.Tensor,
            words: List[str],
            query: str,
            k: int = 10,
    ) -> List[Tuple[str, float]]:
        if query not in words:
            return []
        emb_norm = F.normalize(embeddings, dim=-1)
        qi = words.index(query)
        sims = emb_norm @ emb_norm[qi]
        sims[qi] = -1e4
        top_vals, top_idx = sims.topk(min(k, len(words) - 1))
        return [(words[i], round(v.item(), 4)) for i, v in zip(top_idx.tolist(), top_vals)]

    def side_by_side_knn(
            self,
            trained_emb: torch.Tensor,
            random_emb: torch.Tensor,
            query_words: List[str],
            stratum_label: str,
            k: int = 10,
    ):
        rows: List[Dict] = []
        for q in query_words:
            if q not in self.test_words:
                continue

            trained_nn = self.knn_for_query(trained_emb, self.test_words, q, k)
            random_nn = self.knn_for_query(random_emb, self.test_words, q, k)
            morf_seg = " | ".join(self.word_to_segments.get(q, []))
            morpheus_seg = " | ".join(self.model_segments(self.trained_model, q))

            for rank in range(k):
                t_word, t_sim = trained_nn[rank] if rank < len(trained_nn) else ("", 0.0)
                r_word, r_sim = random_nn[rank] if rank < len(random_nn) else ("", 0.0)
                rows.append({
                    "stratum": stratum_label,
                    "query": q,
                    "rank": rank + 1,
                    "trained_neighbor": t_word,
                    "trained_sim": round(t_sim, 4),
                    "random_neighbor": r_word,
                    "random_sim": round(r_sim, 4),
                    "morfessor_seg": morf_seg,
                    "morpheus_seg": morpheus_seg,
                })

        df = pd.DataFrame(rows)
        out_path = self.output_dir / f"knn_{stratum_label}.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] KNN CSV: {out_path}")
        return out_path

    def per_stratum_coherence(
            self,
            trained_emb: torch.Tensor,
    ) -> pd.DataFrame:
        rows = []

        for stratum_name in ("seen", "oov"):
            words_in_stratum = self.stratified[stratum_name]
            if len(words_in_stratum) < 10:
                continue

            indices = [self.test_words.index(w) for w in words_in_stratum if w in self.test_words]
            sub_emb = trained_emb[indices]
            sub_words = [self.test_words[i] for i in indices]

            coh = root_cluster_coherence(
                word_embeddings=sub_emb,
                words=sub_words,
                word_to_root=self.word_to_root,
                n_neg_samples=2000,
                min_group_size=2,
                seed=self.seed,
            )
            row = {"stratum": stratum_name, "n_words": len(sub_words), **coh}
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "per_stratum_coherence.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] Per-stratum coherence written: {out_path}")
        return df

    def boundary_f1_on_set(
            self,
            words: List[str],
            stratum_label: str,
            threshold: float = 0.5,
    ) -> Dict:
        tp = fp = fn = tn = 0
        per_word_results = []

        for w in words:
            if w not in self.word_to_segments:
                continue
            morf_segs = self.word_to_segments[w]

            morf_bnd = set()
            pos = 0
            for seg in morf_segs[:-1]:
                pos += len(seg)
                morf_bnd.add(pos)

            morpheus_segs = self.model_segments(self.trained_model, w, threshold)
            morph_bnd = set()
            pos = 0
            for seg in morpheus_segs[:-1]:
                pos += len(seg)
                morph_bnd.add(pos)

            n_chars = len(w)
            for i in range(1, n_chars):
                truth = i in morf_bnd
                pred = i in morph_bnd
                if truth and pred:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif truth and not pred:
                    fn += 1
                else:
                    tn += 1

            per_word_results.append({
                "word": w,
                "morfessor": " | ".join(morf_segs),
                "morpheus": " | ".join(morpheus_segs),
                "agree": morf_bnd == morph_bnd,
            })

        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        result = {
            "stratum": stratum_label,
            "n_words": len(words),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        details_path = self.output_dir / f"boundary_details_{stratum_label}.csv"
        pd.DataFrame(per_word_results).to_csv(details_path, index=False)
        return result

    def boundary_f1_all_strata(self) -> pd.DataFrame:
        rows = []
        for stratum in ("seen", "oov", "curated_oov", "nonce"):
            words = self.stratified.get(stratum, [])
            if len(words) < 5:
                continue
            res = self.boundary_f1_on_set(words, stratum)
            rows.append(res)
            global_logger.info(
                f"[PaperEval] Boundary F1 [{stratum}]: "
                f"acc={res['accuracy']} P={res['precision']} R={res['recall']} F1={res['f1']}"
            )

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "boundary_f1_strata.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] Boundary F1 written: {out_path}")
        return df

    def tsne_root_families(
            self,
            trained_emb: torch.Tensor,
            n_root_families: int = 15,
            min_members: int = 4,
    ):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            global_logger.warning("[PaperEval] sklearn missing — skipping t-SNE")
            return

        root_groups: Dict[str, List[int]] = defaultdict(list)
        for i, w in enumerate(self.test_words):
            r = self.word_to_root.get(w)
            if r and r != "<UNK>":
                root_groups[r].append(i)

        eligible_roots = [r for r, idxs in root_groups.items() if len(idxs) >= min_members]
        eligible_roots.sort(key=lambda r: -len(root_groups[r]))
        eligible_roots = eligible_roots[:n_root_families]

        if not eligible_roots:
            global_logger.warning("[PaperEval] No eligible root families — skipping t-SNE")
            return

        selected_idx: List[int] = []
        selected_words: List[str] = []
        selected_roots: List[str] = []
        for r in eligible_roots:
            for idx in root_groups[r][:8]:
                selected_idx.append(idx)
                selected_words.append(self.test_words[idx])
                selected_roots.append(r)

        sub_emb = trained_emb[selected_idx].numpy()

        global_logger.info(
            f"[PaperEval] Running t-SNE on {len(selected_idx)} words "
            f"across {len(eligible_roots)} root families..."
        )
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(selected_idx) // 4),
            random_state=self.seed,
            init="pca",
            learning_rate="auto",
        )
        proj = tsne.fit_transform(sub_emb)

        palette = sns.color_palette("husl", len(eligible_roots))
        root_to_color = {r: palette[i] for i, r in enumerate(eligible_roots)}

        fig, ax = plt.subplots(figsize=(12, 10))
        for i, (x, y) in enumerate(proj):
            r = selected_roots[i]
            ax.scatter(x, y, c=[root_to_color[r]], s=80, alpha=0.7, edgecolors="none")
            ax.annotate(
                selected_words[i],
                (x, y),
                fontsize=7,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )

        legend_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=root_to_color[r], markersize=8, label=r)
            for r in eligible_roots
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True, ncol=2)
        ax.set_title(
            f"Morpheus word embeddings — t-SNE projection by root family "
            f"({len(eligible_roots)} roots, {len(selected_idx)} words)"
        )
        ax.set_xticks([])
        ax.set_yticks([])

        out_path = self.output_dir / "tsne_root_families.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        global_logger.info(f"[PaperEval] t-SNE saved: {out_path}")

    def fertility_comparison(self) -> pd.DataFrame:
        if self.tokenizer_dir is None:
            global_logger.info("[PaperEval] No tokenizer dir — skipping fertility comparison")
            return pd.DataFrame()

        from src.model_development.tokenization.morpheus_tokenizer import MorpheusTokenizer

        global_logger.info(f"[PaperEval] Loading MorpheusTokenizer from {self.tokenizer_dir}")
        tok = MorpheusTokenizer.load(
            path=self.tokenizer_dir,
            morpheus_model=self.trained_model,
            device=self.device,
        )

        rows = []
        for stratum in ("seen", "oov", "curated_oov", "nonce"):
            words = self.stratified.get(stratum, [])
            if not words:
                continue
            total_tokens_morpheus = 0
            total_tokens_morfessor = 0
            total_chars = 0
            for w in words:
                tokens = tok.tokenize(w)
                total_tokens_morpheus += len(tokens)
                morf_segs = self.word_to_segments.get(w, [w])
                total_tokens_morfessor += len(morf_segs)
                total_chars += len(w)

            rows.append({
                "stratum": stratum,
                "n_words": len(words),
                "morpheus_fertility": round(total_tokens_morpheus / max(len(words), 1), 4),
                "morfessor_fertility": round(total_tokens_morfessor / max(len(words), 1), 4),
                "morpheus_compression": round(total_chars / max(total_tokens_morpheus, 1), 4),
                "morfessor_compression": round(total_chars / max(total_tokens_morfessor, 1), 4),
            })

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "fertility_by_stratum.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] Fertility comparison written: {out_path}")
        return df

    def summary(self, coh_df, bnd_df, fert_df, mas_df=None):
        long_rows: List[Dict] = []

        def push(category: str, metric: str, stratum: str, model: str, value):
            long_rows.append({
                "category": category,
                "metric": metric,
                "stratum": stratum,
                "model": model,
                "value": value,
            })

        if not coh_df.empty:
            for _, row in coh_df.iterrows():
                push("coherence", "intra_root_cosine", row["stratum"], "morpheus_trained", row["intra_root_cosine"])
                push("coherence", "inter_root_cosine", row["stratum"], "morpheus_trained", row["inter_root_cosine"])
                push("coherence", "delta", row["stratum"], "morpheus_trained", row["delta"])
                push("coherence", "n_words", row["stratum"], "morpheus_trained", int(row["n_words"]))

        if not bnd_df.empty:
            for _, row in bnd_df.iterrows():
                for metric in ("accuracy", "precision", "recall", "f1"):
                    push("boundary_vs_morfessor", metric, row["stratum"], "morpheus_trained", row[metric])
                push("boundary_vs_morfessor", "n_words", row["stratum"], "morpheus_trained", int(row["n_words"]))

        if not fert_df.empty:
            for _, row in fert_df.iterrows():
                for col in fert_df.columns:
                    if col.endswith("_fertility"):
                        model_name = col.replace("_fertility", "")
                        push("fertility", "tokens_per_word", row["stratum"], model_name, row[col])
                    elif col.endswith("_compression"):
                        model_name = col.replace("_compression", "")
                        push("compression", "chars_per_token", row["stratum"], model_name, row[col])

        if mas_df is not None and not mas_df.empty:
            for _, row in mas_df.iterrows():
                for col in mas_df.columns:
                    if col.endswith("_mas"):
                        model_name = col.replace("_mas", "")
                        push("mas_vs_morfessor", "boundary_agreement_pct", row["stratum"], model_name, row[col])

        long_df = pd.DataFrame(long_rows)
        long_path = self.output_dir / "summary_metrics.csv"
        long_df.to_csv(long_path, index=False)
        global_logger.info(f"[PaperEval] Structured summary CSV: {long_path}")

        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("MORPHEUS PAPER EVALUATION SUMMARY")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        summary_lines.append(f"Checkpoint     : {self.checkpoint_path}")
        summary_lines.append(f"Train vocab    : {len(self.train_vocab):,} words")
        summary_lines.append(f"Test set       : {len(self.test_words)} unique words")
        summary_lines.append(f"  seen         : {len(self.stratified['seen'])}")
        summary_lines.append(f"  oov          : {len(self.stratified['oov'])}")
        summary_lines.append(f"  curated_oov  : {len(self.stratified['curated_oov'])}")
        summary_lines.append(f"  nonce        : {len(self.stratified['nonce'])}")
        summary_lines.append("")

        if not coh_df.empty:
            summary_lines.append("ROOT CLUSTER COHERENCE")
            summary_lines.append("-" * 70)
            for _, row in coh_df.iterrows():
                summary_lines.append(
                    f"  [{row['stratum']:>5}] n={int(row['n_words']):>4} "
                    f"intra={row['intra_root_cosine']:.4f} "
                    f"inter={row['inter_root_cosine']:.4f} "
                    f"delta={row['delta']:.4f}"
                )
            summary_lines.append("")

        if not bnd_df.empty:
            summary_lines.append("BOUNDARY F1 vs MORFESSOR")
            summary_lines.append("-" * 70)
            for _, row in bnd_df.iterrows():
                summary_lines.append(
                    f"  [{row['stratum']:>12}] n={int(row['n_words']):>4} "
                    f"acc={row['accuracy']:.3f} P={row['precision']:.3f} "
                    f"R={row['recall']:.3f} F1={row['f1']:.3f}"
                )
            summary_lines.append("")

        if not fert_df.empty:
            summary_lines.append("FERTILITY (tokens/word) — all tokenizers")
            summary_lines.append("-" * 70)
            fert_cols = [c for c in fert_df.columns if c.endswith("_fertility")]
            for _, row in fert_df.iterrows():
                summary_lines.append(f"  [{row['stratum']:>12}] n={int(row['n_words']):>4}")
                for col in fert_cols:
                    name = col.replace("_fertility", "")
                    summary_lines.append(f"      {name:<20} {row[col]:>6.3f}")
            summary_lines.append("")

        if mas_df is not None and not mas_df.empty:
            summary_lines.append("MAS — boundary agreement with Morfessor (%)")
            summary_lines.append("-" * 70)
            mas_cols = [c for c in mas_df.columns if c.endswith("_mas")]
            for _, row in mas_df.iterrows():
                summary_lines.append(f"  [{row['stratum']:>12}] n={int(row['n_words']):>4}")
                for col in mas_cols:
                    name = col.replace("_mas", "")
                    summary_lines.append(f"      {name:<20} {row[col]:>6.2f}%")
            summary_lines.append("")

        summary_lines.append("Artifacts written to:")
        summary_lines.append(f"  {self.output_dir.absolute()}")
        summary_lines.append("=" * 70)

        text = "\n".join(summary_lines)
        out_path = self.output_dir / "SUMMARY.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        global_logger.info(f"[PaperEval] Summary written: {out_path}")

    def load_classical_tokenizers(self):
        if self.benchmarker_results_dir is None:
            global_logger.info("[PaperEval] No benchmarker_results_dir — skipping classical tokenizers")
            return
        self.classical_tokenizers = discover_classical_tokenizers(
            self.benchmarker_results_dir,
            preferred_vocab=self.preferred_classical_vocab,
        )
        from src.benchmarker.metrics.external_tokenizers import load_turkish_tokenizer_or_none
        tt = load_turkish_tokenizer_or_none()
        if tt is not None:
            self.classical_tokenizers.append(tt)
        if not self.classical_tokenizers:
            global_logger.warning(
                f"[PaperEval] No classical tokenizers found in {self.benchmarker_results_dir}. "
                f"Run `python -m src.benchmarker.benchmarks.classical` first."
            )

    def segmentation_comparison(self) -> pd.DataFrame:
        wide_rows: List[Dict] = []
        long_rows: List[Dict] = []

        all_queries: List[Tuple[str, str]] = []
        for stratum in ("curated_seen", "curated_suffix", "curated_oov", "nonce"):
            for w in self.stratified.get(stratum, []):
                all_queries.append((stratum, w))

        for stratum, word in all_queries:
            morf_segs = self.word_to_segments.get(word, [word])
            morpheus_segs = self.model_segments(self.trained_model, word)

            tokenizer_outputs: Dict[str, List[str]] = {
                "morfessor": morf_segs,
                "morpheus": morpheus_segs,
            }
            for tok in self.classical_tokenizers:
                tokenizer_outputs[tok.name.lower()] = tok.segment(word)

            wide_row = {
                "stratum": stratum,
                "word": word,
                "n_chars": len(word),
            }
            for name, segs in tokenizer_outputs.items():
                wide_row[name] = " | ".join(segs)
                wide_row[f"{name}_n"] = len(segs)
            wide_rows.append(wide_row)

            for name, segs in tokenizer_outputs.items():
                long_rows.append({
                    "stratum": stratum,
                    "word": word,
                    "n_chars": len(word),
                    "tokenizer": name,
                    "segmentation": " | ".join(segs),
                    "n_segments": len(segs),
                })

        wide_df = pd.DataFrame(wide_rows)
        wide_path = self.output_dir / "segmentation_comparison_wide.csv"
        wide_df.to_csv(wide_path, index=False)
        global_logger.info(f"[PaperEval] Segmentation comparison (wide): {wide_path}")

        long_df = pd.DataFrame(long_rows)
        long_path = self.output_dir / "segmentation_comparison_long.csv"
        long_df.to_csv(long_path, index=False)
        global_logger.info(f"[PaperEval] Segmentation comparison (long): {long_path}")

        return wide_df

    def fertility_full_comparison(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for stratum in ("seen", "oov", "curated_oov", "nonce"):
            words = self.stratified.get(stratum, [])
            if not words:
                continue

            total_chars = sum(len(w) for w in words)

            morf_total = 0
            for w in words:
                morf_total += len(self.word_to_segments.get(w, [w]))

            morpheus_total = 0
            for w in words:
                morpheus_total += len(self.model_segments(self.trained_model, w))

            row = {
                "stratum": stratum,
                "n_words": len(words),
                "morfessor_fertility": round(morf_total / max(len(words), 1), 4),
                "morfessor_compression": round(total_chars / max(morf_total, 1), 4),
                "morpheus_fertility": round(morpheus_total / max(len(words), 1), 4),
                "morpheus_compression": round(total_chars / max(morpheus_total, 1), 4),
            }

            for tok in self.classical_tokenizers:
                total = 0
                for w in words:
                    total += len(tok.segment(w))
                row[f"{tok.name.lower()}_fertility"] = round(total / max(len(words), 1), 4)
                row[f"{tok.name.lower()}_compression"] = round(total_chars / max(total, 1), 4)

            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "fertility_full_comparison.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] Full fertility comparison: {out_path}")
        return df

    def mas_against_morfessor(self) -> pd.DataFrame:
        rows: List[Dict] = []

        def boundaries_from_segs(segs: List[str]) -> set:
            bnds = set()
            pos = 0
            for s in segs[:-1]:
                pos += len(s)
                bnds.add(pos)
            return bnds

        for stratum in ("seen", "oov", "curated_oov", "nonce"):
            words = self.stratified.get(stratum, [])
            if len(words) < 5:
                continue

            morpheus_hits = 0
            morpheus_total = 0
            tokenizer_stats = {t.name.lower(): [0, 0] for t in self.classical_tokenizers}

            for w in words:
                ref = boundaries_from_segs(self.word_to_segments.get(w, [w]))
                if not ref:
                    continue

                morph_seg = self.model_segments(self.trained_model, w)
                pred = boundaries_from_segs(morph_seg)
                morpheus_hits += len(ref & pred)
                morpheus_total += len(ref)

                for tok in self.classical_tokenizers:
                    seg = tok.segment(w)
                    pred_t = boundaries_from_segs(seg)
                    tokenizer_stats[tok.name.lower()][0] += len(ref & pred_t)
                    tokenizer_stats[tok.name.lower()][1] += len(ref)

            row = {
                "stratum": stratum,
                "n_words": len(words),
                "morpheus_mas": round(morpheus_hits / max(morpheus_total, 1) * 100, 2),
            }
            for tok_name, (hits, total) in tokenizer_stats.items():
                row[f"{tok_name}_mas"] = round(hits / max(total, 1) * 100, 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "mas_vs_morfessor.csv"
        df.to_csv(out_path, index=False)
        global_logger.info(f"[PaperEval] MAS vs Morfessor: {out_path}")
        return df

    def run_all(self):
        self.load_models()
        self.load_train_vocab()
        self.load_classical_tokenizers()
        self.build_stratified_test_set()

        trained_emb = self.compute_embeddings(self.trained_model, self.test_words)
        random_emb = self.compute_embeddings(self.random_model, self.test_words)

        for stratum in ("curated_seen", "curated_suffix", "curated_oov", "nonce"):
            queries = self.stratified.get(stratum, [])
            if queries:
                self.side_by_side_knn(trained_emb, random_emb, queries, stratum, k=10)

        seg_df = self.segmentation_comparison()
        coh_df = self.per_stratum_coherence(trained_emb)
        bnd_df = self.boundary_f1_all_strata()
        fert_df = self.fertility_full_comparison()
        mas_df = self.mas_against_morfessor()
        self.tsne_root_families(trained_emb)

        self.summary(coh_df, bnd_df, fert_df, mas_df=mas_df)


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent

    checkpoint = str(
        base / "src/model_development/artifacts/checkpoints/turkish_morpheus_a100_best.pt"
    )
    tokenizer_dir = str(base / "src/model_development/artifacts/tokenizers/morpheus_50k")
    morfessor_path = str(base / "src/model_development/artifacts/tokenizers/classical/morfessor_model.bin")
    train_corpus = str(base / "src/model_development/artifacts/datasets/splits/train.txt")
    test_corpus = str(base / "src/model_development/artifacts/datasets/splits/test.txt")
    word_vocab_path = str(base / "src/model_development/artifacts/datasets/splits/word_vocab.pt")
    benchmarker_results = str(base / "src/model_development/artifacts/tokenizers/classical")
    output_dir = str(base / "src/benchmarker/results/paper_eval")

    evaluator = PaperEvaluator(
        checkpoint_path=checkpoint,
        tokenizer_dir=tokenizer_dir,
        morfessor_path=morfessor_path,
        train_corpus_path=train_corpus,
        test_corpus_path=test_corpus,
        word_vocab_path=word_vocab_path,
        benchmarker_results_dir=benchmarker_results,
        preferred_classical_vocab=64000,
        output_dir=output_dir,
        seed=1337,
    )

    evaluator.run_all()
