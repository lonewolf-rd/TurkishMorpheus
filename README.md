# Morpheus: A Morphology-Aware Tokenizer for Turkish

[![arXiv](https://img.shields.io/badge/arXiv-soon-b31b1b.svg)]() [![HuggingFace](https://img.shields.io/badge/🤗-coming_soon-yellow.svg)]() [![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

**Morpheus** is a neural morpheme-aware tokenizer for **Turkish**, an agglutinative language whose semantic content is densely packed into productive suffix chains. It combines unsupervised morphological supervision (Morfessor) with self-supervised objectives (skip-gram negative sampling, root-family contrastive, masked language modeling) to learn segmentations that are simultaneously **morphologically aligned** and **language-modeling-friendly**.

```
evlerimizdekiler  →  ev | leri | miz | deki | ler
                     root  PL+POSS POSS  LOC+REL  PL
                    ("the ones in our houses")
```

Where classical BPE/WordPiece fragment morphologically rich Turkish words into statistically convenient but linguistically opaque subwords, Morpheus produces **interpretable morpheme-level segmentations** while also yielding **structured word embeddings** with strong root-family clustering.

---

## Headline Results

Evaluated on a curated 115 MB monolingual Turkish corpus (informal/academic/news mix), against five baseline tokenizers (BPE, ByteBPE, Unigram, WordPiece, Morfessor) and a fixed 58 M-parameter (param-equalized) GPT-style autoregressive language model:

| Metric | Morpheus | Best baseline | Δ |
|---|---|---|---|
| **BPC (lower better)** | **1.640** | 1.698 (WordPiece) | **−3.5%** |
| Token perplexity | 152 | 523 (WordPiece) | **3.4× lower** |
| **Morphological Alignment Score (MAS) OOV** | **92.4%** | 5.76% (WordPiece) | **16×** |
| Boundary F1 OOV vs Morfessor | **0.916** | — | (segmentation generalization) |
| Root-cluster intra/inter cosine ratio | **274×** | — | (only Morpheus produces embeddings) |
| Peak GPU memory (B=32 generation) | **2,270 MB** | 3,724 MB (classical) | **−39%** |
| Tokenizer artifact size | **0.67 MB** | 2.65 MB (WordPiece) | **4× smaller** |
| End-to-end generation (chars/sec) | 548 | 924 (WordPiece) | −41% (trade-off) |

**Notable**: Morpheus **outperforms its own teacher Morfessor** (BPC 1.640 vs 1.705, −3.9%) — demonstrating that joint training with downstream LM signals refines the morpheme detector beyond pure unsupervised MDL boundaries.

Full results in [Evaluation](#evaluation) section and `src/benchmarker/results/`.

---

## Quick Start

### Install

```bash
git clone https://github.com/<your-org>/TurkishTokenizer-Alpha-v1.git
cd TurkishTokenizer-Alpha-v1

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
- **BPE** (`bpe_{50000,64000}.model`) — SentencePiece BPE
- **ByteBPE** (`byte_bpe_{50000,64000}.model`) — byte-level BPE
- **Unigram** (`unigram_{50000,64000}.model`) — SentencePiece Unigram LM
- **WordPiece** (`wordpiece_{50000,64000}.json`) — HuggingFace `tokenizers` library

**Critical for fairness**: all classical tokenizers train on the *same* corpus that supervises Morpheus.

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
**Output**: `artifacts/checkpoints/turkish_morpheus_a100_v3_best.pt`

Trains the Morpheus neural model with `MorpheusTrainer`. See [Architecture](#architecture) and [Training Recipe](#training-recipe) below. Reported results use the v3 config:

- 22 epochs, batch 256 × grad-accum 2 (effective batch 512), AdamW + cosine LR
- Char dim 320, 3 encoder layers, 4 boundary detector layers, max sentence length 32
- 4 MLM context-encoder layers, 16 SGNS negatives, contrastive temperature 0.10
- TF32 enabled, AMP off (stability over speed)
- ~18 min/epoch on a single A100 80GB

Checkpoints saved every epoch; `_best.pt` tracks lowest validation loss.

### Stage 5: `tokenizer` — discrete tokenizer build

**Input**: trained Morpheus checkpoint, training corpus
**Output**: `artifacts/tokenizers/morpheus_50k/{vocab.json,tokenizer_config.json}`

`build_morpheus_vocab` segments every training word once using Morpheus's hard boundary predictions, accumulates segment frequencies weighted by word count, and selects the top-K (default 50K) most frequent segments plus a small curated set of Turkish suffix templates. The resulting `MorpheusTokenizer` is a standalone, lightweight (~0.7 MB) tokenizer file portable to any downstream pipeline.

### Stage 6: `eval` — intrinsic and morphological evaluation

**Input**: trained checkpoint + all classical tokenizers
**Output**: `src/benchmarker/results/paper_eval/`

Two evaluation suites:
- **`PaperEvaluator`** — stratified test set (seen / OOV / curated_oov / nonce), measures: Boundary F1 vs Morfessor, MAS (morphological alignment), Fertility, Compression, root-cluster Coherence, kNN nearest neighbors per stratum, t-SNE projection of root families.
- **`sigmorphon_eval`** — runs against the SIGMORPHON 2022 Turkish inflection gold set (`data/sigmorphon_tr/tur.gold`), measures lemma-prefix rate, root-in-segments rate, suffix-count accuracy.

### Stage 7 (separate): downstream LM evaluation

```bash
python -m src.benchmarker.benchmarks.lm_eval --mode full
python -m src.benchmarker.benchmarks.lm_eval --mode inference --trained-mode full
```

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
| `L_aux` | Boundary BCE + count MSE vs Morfessor (deep-supervised across detector layers) | Decays geometrically `0.50 → 0.08` over 22 epochs (`decay=0.90`) |
| `L_sgns` | Skip-gram with 16 frequency-weighted negatives, ±6 window, 120K context vocab | Constant `0.7` |
| `L_contrastive` | InfoNCE on root identity (Morfessor's first segment), temperature `0.10` | Constant `0.3` |
| `L_mlm` | Cross-entropy on character autoregressive reconstruction (4 ctx + 2 dec layers) | Constant `1.0` |

The aux schedule realizes a **curriculum**: early epochs anchor on Morfessor (boundary learning); as it decays, distributional signals (SGNS, MLM) take over to shape semantic geometry, with contrastive enforcing morphological consistency throughout.

**Numerical stability**: TF32 enabled for matmul + cuDNN (free 1.5× on A100, FP32-equivalent dynamic range). Loss components computed in FP32 internally to avoid underflow in logsumexp/logsigmoid. AMP/BF16 left off in v3 release for maximum reproducibility.

---

## Evaluation

### Intrinsic — `Stage 6` outputs

Stratified test set: **seen** (800 frequent train-vocab words), **oov** (400 unseen rare), **curated_oov** (10 hand-picked long compounds), **nonce** (5 made-up words).

```
                    Boundary F1     MAS%     Coherence Δ    OOV F1
Morpheus seen       1.000           100.0    0.982          1.000
Morpheus oov        0.916            92.4    0.915          0.916
Morpheus curated    1.000           100.0    —              1.000
Morpheus nonce      0.788            72.2    —              0.788

vs classical (OOV):
WordPiece           —                 5.76   N/A (no emb)   —
Unigram             —                 4.53   N/A            —
BPE                 —                 4.12   N/A            —
```

Full per-stratum breakdown: `src/benchmarker/results/paper_eval/`.

### Downstream language modeling — `Stage 7` outputs

Param-equalized 58 M GPT, 2 epochs over 21 M tokens (BPE) / 14 M tokens (Morpheus, due to higher fertility):

```
                vocab    BPC ↓     tok_ppl ↓    fertility
Morpheus        35,092   1.640     152          1.26
WordPiece       64,000   1.698     523          1.01
Morfessor       30,243   1.705     135          1.31
Unigram         64,000   1.726     344          1.05
BPE             64,000   1.737     397          1.03
ByteBPE         64,000   1.755     383          1.11
```

Morpheus achieves the lowest BPC despite higher fertility — per-token entropy is **3-4× lower** than classical subwords because morpheme tokens are individually more predictable from context, more than compensating for the additional token count.

### Inference

```
                Encode      Gen B=1     Gen B=32   GPU mem    Artifact
                kchar/s     char/s      char/s     B=32 MB    MB
Morpheus        3,885       548         13,843     2,270      0.67
WordPiece       2,100       924         23,497     3,724      2.65
BPE               984       853         21,138     3,724      4.50
Unigram         4,517       823         20,538     3,724      4.91
Morfessor       8,537*      519         12,305     2,022      1.40
ByteBPE           922       835         20,526     3,724      4.44

* warm-cache; cold-start significantly slower
```

**Pareto frontier**: {Morpheus, WordPiece} on (BPC, throughput). {Morpheus, Morfessor} on (BPC, memory). ByteBPE and Unigram dominated by BPE on most axes.

---

## Project Structure

```
.
├── data/
│   ├── corpus_collector_tr/corpus.txt    # local Turkish corpus (vendored)
│   └── sigmorphon_tr/tur.gold            # SIGMORPHON 2022 Turkish gold (test set)
│
├── src/
│   ├── common/                            # shared infrastructure
│   │   ├── configs/                       # single source of truth: main.yaml + logging.yaml
│   │   ├── config_manager.py              # per-configs-dir singleton
│   │   ├── logger.py
│   │   ├── text_utils.py                  # Turkish-aware lower/upper
│   │   └── providers/                     # config_provider, logger_provider
│   │
│   ├── model_development/                 # everything that PRODUCES
│   │   ├── data/                          # preprocessor, analyzer
│   │   ├── model/                         # Morpheus + CharEncoder + BoundaryDetector + SegmentEncoder + MLM head
│   │   ├── training/                      # trainer, dataset, loss, callbacks
│   │   ├── tokenization/                  # classical baselines + MorpheusTokenizer + diagnose
│   │   └── artifacts/                     # produced: datasets, tokenizers, checkpoints
│   │
│   ├── benchmarker/                       # everything that CONSUMES + COMPARES
│   │   ├── metrics/                       # classical + intrinsic + extrinsic
│   │   ├── benchmarks/                    # classical (orchestrator) + paper + sigmorphon + lm_eval
│   │   ├── visualization/
│   │   └── results/                       # paper_eval/ and lm_eval/ outputs
│   │
│   └── run_pipeline.py                    # 6-stage orchestrator
│
└── README.md
```

---

## Corpus

The reported results use a **115 MB curated monolingual Turkish corpus** (~17M words, 261K lines) covering three registers:

| Source | Register | Notes |
|---|---|---|
| **Ekşisözlük** | Informal / colloquial | Rich morphological constructs (`-ymiş`, `-sin`, idiom-heavy) |
| **Dergipark** | Academic / formal | Diverse terminology, derivational morphology |
| **Turkish news sites** | Standard / journalistic | Neutral register, broad vocabulary |

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

- **Turkish NLU / classification tasks**: where token boundaries align with morphemes, downstream classifiers can attend to specific morphological roles (case, tense, person, number) directly via attention.
- **Morphologically-sensitive information retrieval**: stemming via root identification, suffix-aware query expansion.
- **Linguistic research / corpus annotation**: morpheme-level analysis at scale without manual annotation.
- **Pretraining smaller Turkish language models** (≤1B parameters): the lower BPC and structured embeddings give favorable scaling.
- **Educational tools**: visualize Turkish morphology in real-time (e.g. learner apps).
- **Memory-constrained inference**: 39% lower GPU memory than 64K-vocab classical tokenizers — relevant for consumer-GPU and edge deployment.

### When NOT to use Morpheus

- **Real-time generation latency-critical applications**: classical subword tokenizers achieve ~1.7× higher end-to-end character generation throughput (lower fertility = fewer forward passes per character).
- **Multilingual models**: Morpheus is Turkish-specific by design (uses Turkish character vocabulary + Morfessor supervision on Turkish). Use multilingual SentencePiece for cross-lingual tasks.
- **General-purpose LLM pretraining at frontier scale**: for trillion-token pretraining the inductive-bias advantage of morphology likely saturates and standard BPE remains the practical choice.

---

## Status

**v3 (current)** — empirical evaluation complete. Reported results in this README and in `src/benchmarker/results/`. Paper preprint forthcoming on arXiv.

The architectural components (Morpheus model, Poisson-binomial soft segmentation, multi-objective curriculum, evaluation harness, LM benchmark suite) are stable and documented.

### Planned releases
- arXiv preprint (this paper)
- Hugging Face model card + tokenizer release (`morpheus-tr-50k`)
- Hugging Face Spaces demo (interactive segmentation)
- TACL / Cambridge NLP journal submission

### Known limitations
- Single language (Turkish-only by design)
- Single train corpus (115 MB; scaling laws across corpus size not characterized)
- Generation throughput trade-off: ~40% slower chars/sec than BPE in B=1 autoregressive sampling
- Not yet tested as drop-in tokenizer for large pretrained LLMs (Gemma, LLaMA, Mistral) — future work

---

## Citation

```bibtex
@misc{sakar2026morpheus,
  title  = {Morpheus: A Morphology-Aware Tokenizer for Turkish},
  author = {Şakar, Tolga},
  year   = {2026},
  note   = {Preprint forthcoming on arXiv}
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
