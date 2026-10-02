# Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish

[![arXiv](https://img.shields.io/badge/arXiv-Morpheus-b31b1b.svg)](https://arxiv.org/abs/2606.18717) [![HuggingFace](https://img.shields.io/badge/🤗-Open_in_Spaces-yellow.svg)](https://huggingface.co/lonewolflab/Morpheus-TR-50K) [![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

**Morpheus** is a neural morpheme-aware tokenizer **and word embedder** for **Turkish**, an agglutinative language whose semantic content is densely packed into productive suffix chains. It combines unsupervised morphological supervision (Morfessor) with self-supervised objectives (skip-gram negative sampling, root-family contrastive, masked language modeling) to learn segmentations that are simultaneously **morphologically aligned** and **language-modeling-friendly**. Because it is neural, the same forward pass that tokenizes also yields a structured word embedding — so Morpheus is a tokenizer and an embedding model at once.

```
evlerimizdekiler  →  ev | leri | miz | deki | ler
                     root  PL+POSS POSS  LOC+REL  PL
                    ("the ones in our houses")
```

Where classical BPE/WordPiece fragment morphologically rich Turkish words into statistically convenient but linguistically opaque subwords, Morpheus produces **interpretable morpheme-level segmentations** while also yielding **structured word embeddings** with strong root-family clustering.

---

## Headline Results

Morpheus is **exactly reversible on raw Turkish text** — case, punctuation, whitespace, code, emoji — and, among reversible tokenizers, the **only one whose tokens are morphemes**. As an embedder, its frozen 320-d vectors **lead root-family retrieval over 120 gold families (MAP 0.89 vs 0.72 for BGE-M3 and 0.38 for BERTurk)**. All baselines below preserve case and were trained on the same corpus.

| Tokenizer | Inflected words (30,204) | Raw corpus lines (5,000) | Stress set (14) | Main failure mode |
|---|---|---|---|---|
| **Morpheus** | **100%** | **100%** | **14/14** | — (surface-preserving segmentation + byte fallback) |
| Byte-level BPE (GPT-2 style) | **100%** | **100%** | **14/14** | — (operates on raw bytes) |
| BPE / Unigram (SentencePiece, `nmt_nfkc`) | **100%** | 98.7% | 7/14 | NFKC (`²→2`, `…→...`), collapsed or dropped whitespace |
| WordPiece (cased) | **100%** | 24.2% | 3/14 | spacing around punctuation lost |
| TurkishTokenizer (rule-based) | 95.4% | 0% | 0/14 | canonical rewriting (`saatlerde→saatlarda`), inserted spaces |

Byte-level BPE is the only other exactly reversible tokenizer, so the comparison that matters is between the two:

| | Morpheus | Byte-level BPE |
|---|---|---|
| MorphScore macro-F1 ↑ (UD gold) | **0.61** | 0.44 |
| Rare roots segmented exactly ↑ | **70%** | 10% |
| TR-MMLU pure tokens (unique) ↑ | **57.1%** | 32.1% |
| BPC ↓ (58M GPT, equal 10K steps, single seed) | 1.451 | **1.411** |
| Characters per token ↑ | 3.81 | **4.97** |
| Generation, chars/s at batch 1 ↑ | 474 | **874** |
| Tokenizer artifact ↓ | 29.6 MB | **4.6 MB** |
| Word embedding from the same model | ✓ | — |

**What you're choosing:** Morpheus combines exact reversibility with morpheme-aligned units — a combination no other tokenizer in the comparison offers — and adds a root-centric word embedding. The costs are a **30% longer token sequence** than byte-level BPE (1.90 vs 1.56 tokens/word), a heavier tokenizer artifact, and, in a controlled language-modeling probe (58M model, equal compute, where Morpheus also sees 23% less text under the equal-token budget), a **2.8% higher BPC**. Morpheus is the choice when both surface fidelity and morpheme-level units matter; byte-level BPE when sequence length is the binding constraint. Integrating Morpheus into a full Turkish LLM is follow-up work. Full results in [Evaluation](#evaluation) and `src/benchmarker/results/`.

---

## Quick Start

### Install

```bash
git clone https://github.com/<your-org>/TurkishMorpheus.git
cd TurkishMorpheus

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

**Requirements**: Python ≥ 3.13. Training requires CUDA GPU (A100 80GB tested; smaller GPUs need batch_size reduction).

### Provide a Turkish corpus

Place a plain-text Turkish corpus at `data/corpus_collector_tr/corpus.txt` (one document/paragraph per line, UTF-8). The corpus used for the reported results was collected with [**CorpusCollector**](https://github.com/lonewolf-rd/CorpusCollector) — see [Corpus](#corpus) section below.

### End-to-end pipeline

Run all six stages with a single command:

```bash
python -m src.run_pipeline --stage all
```

Or run stages individually (each can resume from previous outputs):

```bash
python -m src.run_pipeline --stage data        # 1. corpus → train/test splits
python -m src.run_pipeline --stage benchmark   # 2. train classical tokenizers + Morfessor
python -m src.run_pipeline --stage dataset     # 3. build sentence caches with Morfessor labels
python -m src.run_pipeline --stage train       # 4. train Morpheus neural model
python -m src.run_pipeline --stage tokenizer   # 5. build 50K MorpheusTokenizer from checkpoint
python -m src.run_pipeline --stage eval        # 6. intrinsic + SIGMORPHON evaluation
```

For downstream language-model evaluation (BPC + inference benchmarks):

```bash
python -m src.benchmarker.benchmarks.lm_eval --mode full
python -m src.benchmarker.benchmarks.lm_eval --mode inference --trained-mode full
```

---

## Pipeline Stages — Detailed

### Stage 1: `data` — corpus ingest + train/test split

**Input**: `data/corpus_collector_tr/corpus.txt`
**Output**: `src/model_development/artifacts/datasets/{raw,splits}/`

- `DataPreprocessor` copies the local corpus to `artifacts/datasets/raw/corpus.txt` (idempotent — skips if size matches)
- `DatasetAnalyzer` prints quantitative stats (line/word/char counts, Turkish character frequencies, morphological density) and creates a 95/5 train/test split using a deterministic seed

**Outputs produced**:
```
artifacts/datasets/raw/corpus.txt           # local corpus (vendored)
artifacts/datasets/splits/train.txt         # 95% of lines, shuffled
artifacts/datasets/splits/test.txt          # 5% held-out
```

### Stage 2: `benchmark` — classical tokenizer training

**Input**: `artifacts/datasets/splits/train.txt`
**Output**: `src/model_development/artifacts/tokenizers/classical/`

Trains five baseline tokenizers on the same training corpus to enable fair comparison:

- **Morfessor** (`morfessor_model.bin`) — unsupervised morphology baseline, 20 batch + 5 online epochs at `corpusweight=1.0`
- **BPE** (`bpe_{50000,64000}.model`) — SentencePiece BPE, `nmt_nfkc` normalization (case preserved), byte fallback
- **ByteBPE** (`byte_bpe_{50000,64000}.json`) — true byte-level BPE (HuggingFace `tokenizers`, GPT-2 style ByteLevel pre-tokenizer/decoder, no normalizer — lossless on any text)
- **Unigram** (`unigram_{50000,64000}.model`) — SentencePiece Unigram LM, `nmt_nfkc` (case preserved), byte fallback
- **WordPiece** (`wordpiece_{50000,64000}.json`) — HuggingFace `tokenizers`, cased (NFC only; no lowercasing or accent stripping)

**Critical for fairness**: all classical tokenizers train on the *same* training split, and none of them folds case, so every language model below predicts the same cased text. (Morfessor is reused if `morfessor_model.bin` exists, since it is also Morpheus's boundary teacher.)

### Stage 3: `dataset` — sentence cache with Morfessor labels

**Input**: `artifacts/datasets/splits/train.txt`, `artifacts/tokenizers/classical/morfessor_model.bin`
**Output**: `artifacts/datasets/splits/{train,test}_sentences.pt`, `{word,root}_vocab.pt`

`build_sentence_cache` pre-tokenizes each sentence into:
- character IDs (per-word, padded to `max_word_len=32`)
- case flags (per-character, for case-aware lowercase recovery)
- Morfessor boundary labels (per-word, `(max_word_len−1)` binary vector)
- per-word word IDs (against a 120K word vocab)
- per-word root IDs (against a 30K root vocab; root = first Morfessor segment)
- sentence attention mask

This is computed once and cached as a `torch.save`-able tensor batch, eliminating per-epoch Morfessor inference overhead.

### Stage 4: `train` — Morpheus neural training

**Input**: sentence caches from Stage 3, Morfessor model from Stage 2
**Output**: `artifacts/checkpoints/turkish_morpheus_a100_v3_final.pt`

Trains the Morpheus neural model with `MorpheusTrainer`. See [Architecture](#architecture) and [Training Recipe](#training-recipe) below. Reported results use the v4 config:

- 10 epochs, batch 256 × grad-accum 2 (effective batch 512), AdamW + cosine LR, `aux_weight_decay=0.80` (lambda 0.50→0.08 over 10 epochs)
- Sentence cache capped at 900K (train) / 100K (val); word vocab still built from the full enlarged corpus
- Boundary labels = Morfessor **root-corrected by Kalbur** (hybrid teacher; training-only, position-based)
- Char dim 320, 3 encoder layers, 4 boundary detector layers, max sentence length 32
- 4 MLM context-encoder layers, 16 SGNS negatives, contrastive temperature 0.10
- TF32 enabled, AMP off (stability over speed); ~30 min/epoch on a single A100 80GB (~5h total)

Checkpoints saved every epoch; `_best.pt` tracks lowest validation loss.

### Stage 5: `tokenizer` — discrete tokenizer build

**Input**: trained Morpheus checkpoint, training corpus
**Output**: `artifacts/tokenizers/morpheus_50k/{vocab.json,tokenizer_config.json}`

`build_morpheus_vocab` segments every training word once using Morpheus's hard boundary predictions, accumulates segment frequencies weighted by word count (word-initial and word-internal segments, plus spaced punctuation, counted exactly as the tokenizer emits them), and selects the top-K (default 50K) most frequent segments plus a small curated set of Turkish suffix templates. 256 byte-fallback tokens and whitespace tokens (`<NL>`, `<TAB>`, `<CR>`) are reserved inside the budget, so `decode(encode(x)) == x` holds for any input text and no input maps to `<UNK>`. The vocabulary is a small JSON file; segmenting unseen words still requires the Morpheus checkpoint.

### Stage 6: `eval` — intrinsic and morphological evaluation

**Input**: trained checkpoint + all classical tokenizers
**Output**: `src/benchmarker/results/paper_eval/`

Two evaluation suites:
- **`PaperEvaluator`** — stratified test set (seen / OOV / curated_oov / nonce), measures: Boundary F1 vs Morfessor, MAS (morphological alignment), Fertility, Compression, root-cluster Coherence, kNN nearest neighbors per stratum, t-SNE projection of root families.
- **`sigmorphon_eval`** — runs against the SIGMORPHON 2022 Turkish inflection gold set (`data/sigmorphon_tr/tur.gold`), measures lemma-prefix rate, root-in-segments rate, suffix-count accuracy.

### Stage 7 (separate): downstream LM evaluation

```bash
python -m src.benchmarker.benchmarks.lm_eval --mode full
python -m src.benchmarker.benchmarks.lm_eval --mode full --seed 1
python -m src.benchmarker.benchmarks.lm_eval --mode full --seed 2
python -m src.benchmarker.benchmarks.lm_eval --mode inference --trained-mode full
```

Two protocols, same model and schedule shape:

- `--mode full` — **equal compute (main protocol)**: every tokenizer trains for exactly 10K optimizer steps of 32 × 512 tokens; test BPC is scored on the final weights. Equal steps means equal tokens processed, not equal epochs: high-fertility tokenizers cover the same text fewer times.
- `--mode long` — **optional diagnostic**: up to 40K steps, stopping 3 evals (every 1K steps) after the best validation BPC unless it improves by ≥ 0.001; test BPC is scored on the best-validation weights. Training lengths then differ per tokenizer, so this is not an equal-budget comparison.

Morpheus encodes a corpus by first segmenting every distinct word in large GPU batches (`MorpheusTokenizer.warm_cache`) and then encoding lines from that cache. On Colab, keep `artifacts/lm_eval_cache/` on Google Drive between sessions — cache files are keyed by tokenizer content, so they are reused only while the tokenizer is unchanged.

In both, intermediate evals use a **validation split** — the `--val-lines` (default 20K) training-file lines right after the ones trained on — and the test split is scored exactly once, so no choice is made on test data. Seed 0 writes to `lm_eval/<mode>/`, other seeds to `lm_eval/<mode>_seed<N>/`; every run refreshes `lm_eval/<mode>_seeds_summary.csv` (BPC mean ± std per tokenizer). Token-stream caches in `artifacts/lm_eval_cache/` are keyed by a hash of each tokenizer's files, so retrained tokenizers are never served stale streams.

Trains a fixed 58 M-parameter (param-equalized) GPT with each of the six tokenizers on the same corpus, measures:
- **BPC** (Bits Per Character) — the headline LM quality metric, normalized by character count so it's directly comparable across tokenizers
- **Per-token perplexity** — secondary
- **Encoding throughput** — chars/sec for the tokenizer alone
- **End-to-end generation throughput** — chars/sec when running autoregressive sampling (batch sizes 1, 8, 32)
- **Peak GPU memory** — during generation
- **Tokenizer artifact size** — on-disk

Outputs to `src/benchmarker/results/lm_eval/{full,inference}/`.

---

## Architecture

```
char_ids, case_flags
        │
        ▼
   CharEncoder              Char + case embeddings → MultiScaleCNN (kernels 2..6)
        │                   → 3 × full self-attention with RoPE (dim=320, heads=5)
        │                   Output: (B, L=32, 320) context-aware char vectors
        ▼
  BoundaryDetector          4 × RoPE attention with adjacent-pair scoring head
        │                   Deep-supervised aux loss (BCE + count regularization)
        │                   vs Morfessor labels with depth-weighted schedule
        │                   Output: boundary_probs ∈ [0,1] of shape (B, L−1=31)
        ▼
  SegmentEncoder            Poisson-binomial DP → soft segment membership (B, L, S=12)
        │                   Char-level attention pooling per segment → segment vectors
        │                   Mean over valid segments + 2-layer FFN + LayerNorm
        ▼
   word_embedding ∈ ℝ³²⁰ (B, 320)
```

### Soft segmentation via Poisson-binomial dynamic programming

Given boundary probabilities `p_i ∈ [0,1]` between adjacent characters, the probability that character `i` belongs to segment `k` is the probability of observing exactly `k` boundaries before position `i` — a Poisson-binomial distribution computed via:

```
f[i, k] = f[i−1, k] · (1 − p_i) + f[i−1, k−1] · p_i
```

This gives a **differentiable** membership matrix that converges to one-hot segment assignments as `p_i → {0, 1}`, recovering hard segmentation at inference time without an architectural switch (model.training flag controls behavior).

### Position-aware boundary prediction

Both `CharEncoder.LocalSelfAttention` and `BoundaryDetector.BoundaryAttention` apply Rotary Position Embedding (RoPE) on the per-head subspace. This is motivated by the fact that morpheme identity in Turkish depends on **position relative to the root** (e.g. the third suffix slot is structurally constrained to host certain morpheme types). Encoding this relative offset directly is more sample-efficient than recovering it from distributional evidence alone.

### Case as a side channel

Rather than doubling the character vocabulary across uppercase/lowercase pairs, we lowercase the input (Turkish-aware: `İ`→`i`, `I`→`ı`) and add a learned 2×16 case-flag embedding concatenated to the character embedding. This halves the embedding rows while keeping morphologically equivalent forms (e.g. `İstanbul` vs `istanbul`) in the same orbit of embedding space.

### Word MLM head (auxiliary semantic objective)

A 4-layer transformer encoder operates over word embeddings within a sentence; 20% of words are replaced with a learnable `[MASK]` token. For each masked position, a 2-layer transformer decoder generates the original word character-by-character, conditioned on the masked position's context vector. The cross-entropy on character predictions provides a vocabulary-free reconstruction signal that complements the discrete SGNS objective.

---

## Training Recipe

Total loss is a weighted sum:

```
L = w_aux · L_aux + w_sgns · L_sgns + w_ctr · L_contrastive + w_mlm · L_mlm
```

| Loss | Role | Weight schedule |
|---|---|---|
| `L_aux` | Boundary BCE + count MSE vs **Morfessor labels, root-corrected by Kalbur** (deep-supervised across detector layers) | Decays geometrically `0.50 → 0.08` over 10 epochs (`decay=0.80`) |
| `L_sgns` | Skip-gram with 16 frequency-weighted negatives, ±6 window, 120K context vocab | Constant `0.7` |
| `L_contrastive` | InfoNCE on root identity (Morfessor's first segment), temperature `0.10` | Constant `0.3` |
| `L_mlm` | Cross-entropy on character autoregressive reconstruction (4 ctx + 2 dec layers) | Constant `1.0` |

**Hybrid teacher (v4):** boundary labels come from **Morfessor** (full coverage incl. OOV, probabilistic) and are **root-corrected by Kalbur** — for dictionary words, intra-root Morfessor boundaries are removed (only when Morfessor agrees on the root end), reducing root over-segmentation. Kalbur is **training-only and position-based** (it never normalizes strings), so Morpheus stays purely neural and surface-preserving at inference. This is categorically different from rule-based tokenizers that apply normalization at runtime.

The aux schedule realizes a **curriculum**: early epochs anchor on the (corrected) teacher; as it decays, distributional signals (SGNS, MLM) take over to shape semantic geometry, with contrastive enforcing morphological consistency throughout — so the model becomes teacher-free and generalizes to OOV.

**Numerical stability**: TF32 enabled for matmul + cuDNN (free 1.5× on A100, FP32-equivalent dynamic range). Loss components computed in FP32 internally to avoid underflow in logsumexp/logsigmoid. AMP/BF16 left off for maximum reproducibility.

---

## Evaluation

All tokenizers are compared on the same Turkish corpus and held-out gold sets. Every baseline preserves case: BPE and Unigram use SentencePiece `nmt_nfkc`, byte-level BPE is GPT-2 style with no normalizer, WordPiece is cased (NFC only), and the Morfessor baseline is retrained on the same corpus. Full CSVs in `src/benchmarker/results/`.

### Reversibility (`roundtrip_eval`) — the LLM gate

Exact reconstruction `decode(encode(x)) == x` at three levels: 30K lowercase inflected wordforms (UD_Turkish-Kenet), the first 5,000 raw lines of the held-out split, and a 14-line stress set (punctuation, casing, numbers, URLs, indented code, tabs/CRLF, irregular spacing, non-Turkish letters, emoji, out-of-vocabulary symbols). See the table in [Headline Results](#headline-results). Run with `python -m src.benchmarker.benchmarks.roundtrip_eval --mode all`.

### Gold morphology — MorphScore (UD_Turkish-Kenet, 30K words) and SIGMORPHON 2022 (856 words)

| Model | MorphScore recall | precision | macro-F1 | SIGM. lemma-prefix | SIGM. root-in-segs |
|---|---|---|---|---|---|
| TurkishTokenizer (not reversible) | 0.760 | **0.564** | **0.648** | 0.711 | **0.633** |
| **Morpheus** | 0.677 | 0.552 | 0.608 | **0.762** | 0.481 |
| Morfessor | 0.651 | 0.477 | 0.550 | 0.708 | 0.349 |
| Byte-level BPE | 0.536 | 0.375 | 0.441 | 0.617 | 0.167 |
| Unigram / BPE / WordPiece | 0.37–0.40 | 0.32–0.33 | 0.35–0.36 | 0.59–0.61 | 0.42–0.43 |

Morpheus is the strongest reversible tokenizer on gold morphology. Byte-level BPE's MorphScore comes from recall rather than precision: splits inside multi-byte characters create many boundaries, some of which coincide with morpheme boundaries.

### Qualitative surface fidelity (50 OOV-leaning words, pure surface match)

The gap between **len%** (cut at the right boundary positions) and **exact%** (token strings exactly match the surface morphemes) quantifies surface rewriting:

| Tokenizer | root% | len% (boundaries) | exact% (surface strings) | drop |
|---|---|---|---|---|
| **Morpheus** | **66** | 38 | **38** | **0** |
| Morfessor | 58 | 28 | 28 | 0 |
| WordPiece | 32 | 22 | 22 | 0 |
| Unigram / BPE | 30–32 | 16–18 | 16–18 | 0 |
| Byte-level BPE | 24 | 6 | 6 | 0 |
| TurkishTokenizer | 64 | **78** | 10 | **68 (canonical rewriting)** |

By phenomenon (len% / exact%), Morpheus's lead is concentrated on **rare and technical roots**; stem alternations remain hard for every surface-preserving tokenizer:

| Phenomenon (n) | Morpheus | Morfessor | Best subword | TurkishTokenizer |
|---|---|---|---|---|
| Rare root (20) | **70 / 70** | 40 / 40 | 45 / 45 | 90 / 20 |
| Derivation (10) | 30 / 30 | **40 / 40** | 20 / 20 | 80 / 10 |
| Consonant softening (7) | **14 / 14** | 0 / 0 | 0 / 0 | 100 / 0 |
| Vowel drop (6) | 17 / 17 | **33 / 33** | 0 / 0 | 0 / 0 |
| Loanword harmony exception (7) | 0 / 0 | 0 / 0 | 0 / 0 | 86 / 0 |

### Token validity — TR-MMLU (Kalbur validator)

| Tokenizer | %TR (unique) | %Pure (unique) | %Pure (freq.-weighted) | Fertility (tok/word) |
|---|---|---|---|---|
| TurkishTokenizer (not reversible) | 79.4 | **65.5** | **78.2** | 1.98 |
| **Morpheus** | 73.0 | 57.1 | 76.5 | 1.90 |
| Morfessor | 66.5 | 46.5 | 75.9 | 1.82 |
| Subword (BPE / Unigram / WordPiece / byte-level BPE) | 81.7–88.5 | 32–33 | 48–51 | **1.48–1.56** |

### Downstream language modeling — BPC

A param-equalized 58 M GPT is trained with each tokenizer for an **identical 10,000 optimizer steps** (equal compute: 164M tokens), single seed. Intermediate evaluations use a validation split of held-out training lines; the test split is scored once on the final weights.

| Tokenizer | BPC ↓ | Raw-text roundtrip | Chars / token | NLL / token |
|---|---|---|---|---|
| WordPiece | 1.380 | 24.2% | 4.97 | 4.75 |
| TurkishTokenizer | 1.407 | 0% | 3.09 | 3.01 |
| Byte-level BPE | **1.411** (best reversible) | **100%** | 4.97 | 4.86 |
| Unigram | 1.416 | 98.7% | 4.59 | 4.51 |
| BPE | 1.425 | 98.7% | 4.60 | 4.54 |
| Morfessor | 1.430 | — | 4.26 | 4.22 |
| **Morpheus** | 1.451 | **100%** | 3.81 | 3.83 |

The two lowest values belong to non-reversible tokenizers whose token streams omit information (WordPiece: spacing around punctuation; TurkishTokenizer: allomorph identity). Between the two exactly reversible tokenizers, byte-level BPE is 0.040 BPC (2.8%) better; under the equal-token budget Morpheus covers 624M characters of training text against 815M, and both curves are still descending at 10K steps with similar slopes. Earlier versions reported a lower Morpheus BPC (1.425); that run's pre-tokenizer dropped punctuation from Morpheus's token stream and the baselines folded case.

### Efficiency

| | Morpheus | Byte-level BPE | BPE | TurkishTokenizer |
|---|---|---|---|---|
| Generation chars/s (B=1 / B=32) | 474 / 12,378 | **874 / 21,787** | 827 / 20,672 | 389 / 10,083 |
| Peak GPU mem (B=32 generation) | 3,021 MB | 3,724 MB | 3,724 MB | **2,152 MB** |
| Encode speed (chars/s) | 1.51M | 1.53M | 0.98M | **5.52M** |
| Decode speed (words/s) | 0.44M | 0.62M | 0.32M | **0.81M** |
| Tokenizer artifact | 29.6 MB | 4.6 MB | **1.3 MB** | — |

The GPU-memory differences follow vocabulary size (50K for Morpheus, 64K for the subword baselines, 32K for TurkishTokenizer), not morphology.

### Word embeddings — Morpheus vs BERTurk vs BGE-M3

Because Morpheus is neural, the same forward pass that tokenizes also yields a word embedding. We evaluate these **frozen** vectors against BERTurk (768-d) and the multilingual retriever BGE-M3 (1024-d), both encoding isolated words. Retrieval and verification use 120 root families (1,664 words) sampled from UD_Turkish-Kenet gold stems.

| Task | Morpheus (320) | BERTurk (768) | BGE-M3 (1024) |
|---|---|---|---|
| Root-family retrieval (MAP ↑) | **0.89** | 0.38 | 0.72 |
| Same-root verification (ROC-AUC ↑) | **0.999** | 0.794 | 0.951 |
| Number probing (acc ↑; majority class 0.52) | 0.59 | **0.95** | 0.91 |
| Case probing (acc ↑; majority class 0.20) | 0.22 | **0.89** | 0.81 |
| WikiANN-tr NER (macro-F1 ↑) | 0.48 | **0.79** | 0.76 |

This is a **deliberate, architectural trade-off**: the root-identity contrastive objective pulls a root's inflections together — sharpening root geometry (hence the retrieval/verification wins) while collapsing the inflectional contrasts a probe reads — and the static per-word vector lacks the sentence context NER needs. Morpheus is therefore **complementary** to contextual encoders: well suited to the **lexical index** of a multi-vector RAG system, paired with a dense semantic encoder for context. Full results in `src/benchmarker/results/paper_eval/embeddings/`.

---

## Corpus

The reported results use a curated monolingual Turkish corpus (~17M words, 261K lines) **enlarged with cleaned Turkish Wikipedia** (`src/model_development/data/wikipedia_ingest.py` — TR-alphabet/stopword/length/markup filtering + dedup), covering four registers:

| Source | Register | Notes |
|---|---|---|
| **Ekşisözlük** | Informal / colloquial | Rich morphological constructs (`-ymiş`, `-sin`, idiom-heavy) |
| **Dergipark** | Academic / formal | Diverse terminology, derivational morphology |
| **Turkish news sites** | Standard / journalistic | Neutral register, broad vocabulary |
| **Turkish Wikipedia (v4)** | Encyclopedic | Broad vocabulary + word-form diversity; aggressively cleaned/filtered |

The corpus was collected and parsed with the companion repository:

### 🔗 [**CorpusCollector**](https://github.com/lonewolf-rd/CorpusCollector)

A standalone scraping + preprocessing toolkit that documents:
- Source URLs, scraping protocol, rate-limiting policy
- Per-source extraction logic (HTML stripping, URL removal, Unicode normalization)
- License/ethical considerations per source
- Deduplication and length-filtering scripts

This separation enables **full reproducibility**: anyone can recreate an equivalent corpus by re-running CorpusCollector with the documented configurations. The frozen corpus used in the paper will be released on Hugging Face Datasets alongside its SHA-256 hash for exact reproduction.

**For your own use**: drop any UTF-8 Turkish text file into `data/corpus_collector_tr/corpus.txt` and the pipeline will train on your data.

---

## Use Cases

Morpheus is designed for applications where **morphological structure**, **interpretable token boundaries**, or **embedding quality** are valuable. Concrete recommended uses:

### When to use Morpheus

- **Faithful text pipelines**: anywhere decoded text must match the input byte for byte (case, punctuation, whitespace, code, symbols) and morpheme structure is useful.
- **Morphologically-sensitive information retrieval**: stemming via root identification, suffix-aware query expansion, the lexical index of a multi-vector RAG system.
- **Linguistic research / corpus annotation**: morpheme-level analysis at scale without manual annotation; strongest on rare and technical roots.
- **Educational tools**: visualize Turkish morphology in real-time (e.g. learner apps).
- **Memory-constrained inference**: ~19% lower GPU memory than 64K-vocab classical tokenizers — relevant for consumer-GPU and edge deployment.

### When NOT to use Morpheus

- **Throughput-bound generation**: byte-level BPE generates ~1.8× more characters per second at the tested scale (fewer tokens = fewer forward passes per character).
- **When BPC at a fixed token budget is the deciding metric**: byte-level BPE is 0.040 BPC lower at 58M parameters / 10K steps.
- **Lookup-table deployment**: Morpheus needs its ~30 MB neural model to segment unseen words.
- **Multilingual models**: Morpheus is Turkish-specific by design (uses Turkish character vocabulary + Morfessor supervision on Turkish). Use multilingual SentencePiece for cross-lingual tasks.
- **General-purpose LLM pretraining at frontier scale**: for trillion-token pretraining the inductive-bias advantage of morphology likely saturates and standard BPE remains the practical choice.

---

## Status

**v4 (current)** — trained on a Turkish corpus enlarged with cleaned Turkish Wikipedia, with Kalbur root-coherence label correction and a full lossless-vs-lossy comparison against TurkishTokenizer and the subword family. The tokenizer (v2) preserves all text — punctuation, case, whitespace — and adds byte fallback, so raw-text reconstruction is exact; all results were re-run with it. Reported results in this README, in `src/benchmarker/results/`, and in the [arXiv paper](https://arxiv.org/abs/2606.18717).

The architectural components (Morpheus model, Poisson-binomial soft segmentation, multi-objective curriculum, hybrid Morfessor+Kalbur teacher, evaluation harness incl. reversibility / MorphScore / SIGMORPHON / qualitative surface-fidelity / LM-BPC suites) are stable and documented.

### Planned releases
- arXiv preprint (this paper)
- Hugging Face model card + tokenizer release (`lonewolflab/Morpheus-TR-50K`)
- Hugging Face Spaces demo (interactive segmentation + embedding explorer)
- TACL / Cambridge NLP journal submission

### Trade-offs to weigh (not blockers)

The points below are the engineering trade-offs to weigh when adopting Morpheus as a tokenizer or embedder — none of them is a correctness blocker:

- **Fertility** is higher than subword tokenizers (1.90 vs 1.48–1.56 tokens/word) — the deliberate cost of morpheme-level tokenization, paid back in morphological structure, not in BPC. Integrating Morpheus into a full Turkish LLM (equal-text, multi-seed, ≥1B-parameter comparisons with byte-level BPE) is the subject of a follow-up paper.
- **Suffix chains and stem alternations:** the boundary detector often merges adjacent suffixes (`rol | lerde`, `demir | cilik`) and struggles with consonant softening and vowel drop (`yap | rağı | n`). Rule-based dictionary tokenizers place such boundaries better on in-dictionary words — but they pay for it with lossy decoding. Closing this gap is the focus of the next iteration.
- **Vocabulary headroom:** a reversible morpheme-merge layer (frequent root+suffix combos → single tokens) can cut fertility/BPC further *without* surface loss — a planned, drop-in improvement, not a redesign.
- **Scope:** Turkish-specific by design. Drop-in use with frontier LLMs (Gemma/LLaMA) and downstream benchmarks (NER, STSb-TR, TurBLiMP) are planned next steps, not current claims.

---

## Citation

```bibtex
@misc{sakar2026morpheus,
  title  = {Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish},
  author = {Şakar, Tolga},
  year   = {2026},
  note   = {arxiv.org/abs/2606.18717}
}
```

Related prior work by the same author (RAG efficiency in NLP):

```bibtex
@article{sakar2025rag,
  title   = {Maximizing {RAG} efficiency: A comparative analysis of {RAG} methods},
  author  = {Şakar, Tolga and Emekci, Hakan},
  journal = {Natural Language Processing},
  volume  = {31},
  number  = {1},
  year    = {2025},
  publisher = {Cambridge University Press}
}
```

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

- **Morfessor** (Creutz & Lagus, 2002, 2007) as unsupervised morphological supervisor and reference baseline.
- **SentencePiece** (Kudo & Richardson, 2018) and **HuggingFace tokenizers** for the BPE / Unigram / WordPiece baselines.
- **SIGMORPHON 2022 Turkish task** organizers for the inflection gold standard used in morphological evaluation.
- The Turkish NLP community for prior work on morphologically-aware processing (BERTurk, TURNA, Zemberek, TRMorph) that motivated this study.
