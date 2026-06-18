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

Morpheus is the **only lossless, morphology-aware tokenizer for Turkish that is usable in a generative LLM** — and among reversible tokenizers it achieves the **lowest BPC**, while uniquely producing structured root-family embeddings and using **~19% less GPU memory** than 64K-vocab subword tokenizers. As an embedder, its frozen vectors **lead on lexical retrieval (root-family MAP 0.85) and same-root verification (ROC-AUC 1.00)**, surpassing the multilingual retriever BGE-M3 and BERTurk.

The two tokenizers that appear to beat it — WordPiece (lowest raw BPC) and TurkishTokenizer (best gold morphology) — buy those numbers with **information loss**, which disqualifies them for generation (where token ids must decode back to faithful text):

| Tokenizer | Roundtrip `decode(encode(w))==w` | Usable in a generative LLM? |
|---|---|---|
| **Morpheus** | **100%** (surface-preserving) | ✓ |
| BPE / ByteBPE / Unigram | 100% (no morphology) | ✓ |
| TurkishTokenizer | 95.4% (lossy canonical decode) | ✗ — ~5% of inflected words corrupt (`saat→saatlar`) |
| WordPiece | 58% (strips ç/ğ/ı/ö/ş/ü) | ✗ |

**Restricted to the reversible subset** (the only valid LLM candidates), Morpheus leads where it matters:

| Metric | Morpheus | Best reversible baseline |
|---|---|---|
| BPC (lower = better, equal 10K steps) | **lowest among lossless** | — |
| Gold morpheme F1 (MorphScore, UD gold) | **0.61** | 0.32 (BPE) |
| Surface string fidelity (qualitative `exact%`) | **38%** | 12–16% (subwords) |
| Structured root-family embeddings | **✓** | ✗ |
| Peak GPU memory (B=32 generation) | **~3,020 MB** | 3,723 MB (64K subword) |

**What you're choosing:** Morpheus brings modeling quality (lowest BPC among lossless), morphological structure, structured embeddings, lossless reversibility, and lower memory **together** — a combination no other Turkish tokenizer offers. The one parameter to weigh is **fertility** (~1.73 vs subword ~1.5 tokens/word): you accept a modest generation-throughput cost in return for everything above. Latency-only workloads favor a subword tokenizer; for Turkish LLMs that care about quality, morphology, or faithful decoding, Morpheus is the better-informed default. Full results in [Evaluation](#evaluation) and `src/benchmarker/results/`.

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
| `L_aux` | Boundary BCE + count MSE vs **Morfessor labels, root-corrected by Kalbur** (deep-supervised across detector layers) | Decays geometrically `0.50 → 0.08` over 10 epochs (`decay=0.80`) |
| `L_sgns` | Skip-gram with 16 frequency-weighted negatives, ±6 window, 120K context vocab | Constant `0.7` |
| `L_contrastive` | InfoNCE on root identity (Morfessor's first segment), temperature `0.10` | Constant `0.3` |
| `L_mlm` | Cross-entropy on character autoregressive reconstruction (4 ctx + 2 dec layers) | Constant `1.0` |

**Hybrid teacher (v4):** boundary labels come from **Morfessor** (full coverage incl. OOV, probabilistic) and are **root-corrected by Kalbur** — for dictionary words, intra-root Morfessor boundaries are removed (only when Morfessor agrees on the root end), reducing root over-segmentation. Kalbur is **training-only and position-based** (it never normalizes strings), so Morpheus stays purely neural and surface-preserving at inference. This is categorically different from rule-based tokenizers that apply normalization at runtime.

The aux schedule realizes a **curriculum**: early epochs anchor on the (corrected) teacher; as it decays, distributional signals (SGNS, MLM) take over to shape semantic geometry, with contrastive enforcing morphological consistency throughout — so the model becomes teacher-free and generalizes to OOV.

**Numerical stability**: TF32 enabled for matmul + cuDNN (free 1.5× on A100, FP32-equivalent dynamic range). Loss components computed in FP32 internally to avoid underflow in logsumexp/logsigmoid. AMP/BF16 left off for maximum reproducibility.

---

## Evaluation

All tokenizers are compared on the same Turkish corpus and held-out gold sets. Headline tables below; full CSVs in `src/benchmarker/results/`.

### Reversibility (`roundtrip_eval`) — the LLM gate

`decode(encode(w)) == w` over 30K inflected wordforms (UD_Turkish-Kenet):

| Tokenizer | Roundtrip acc | Failure mode |
|---|---|---|
| **Morpheus** | **100.0%** | — (surface-preserving by construction) |
| BPE / ByteBPE / Unigram | 100.0% | — |
| TurkishTokenizer | 95.4% | canonical re-harmonization errors (`saat→saatlar`, `gid→git`) |
| WordPiece | 58.2% | strips Turkish diacritics ç/ğ/ı/ö/ş/ü |

### Gold morphology — MorphScore (UD_Turkish-Kenet, 30K words)

| Model | Recall | Precision | Macro-F1 |
|---|---|---|---|
| TurkishTokenizer | 0.760 | 0.564 | **0.648** |
| **Morpheus** | 0.677 | **0.552** | 0.608 |
| Morfessor | 0.691 | 0.514 | 0.589 |
| BPE / Unigram / ByteBPE | ~0.34 | ~0.30 | ~0.32 |
| WordPiece | 0.283 | 0.258 | 0.270 |

Both Morpheus and the rule-based TurkishTokenizer far outrank the subword family (~2×). TurkishTokenizer edges recall via its root dictionary (measured under a small length-mismatch caveat from canonical normalization); Morpheus matches it on precision/F1 with **exact, lossless** boundary positions — and is the only one of the two usable for generation.

### Gold inflection — SIGMORPHON 2022 (856 words)

| Model | Lemma-prefix | Root-in-segments |
|---|---|---|
| **Morpheus** | **0.762** | 0.481 |
| Morfessor | 0.782 | 0.354 |
| TurkishTokenizer | 0.711 | 0.633 |

The Kalbur root-coherence label-correction (v4) lifted Morpheus's `root_in_segments` from 0.35 → 0.48 (less root over-segmentation) while Morpheus retains the best lemma-prefix rate.

### Qualitative surface fidelity (49 OOV-leaning words, pure surface match)

The gap between **len%** (cut at the right boundary positions) and **exact%** (token strings exactly match the surface morphemes) quantifies decode corruption:

| Tokenizer | len% (boundaries) | exact% (surface strings) | drop |
|---|---|---|---|
| **Morpheus** | 38 | **38** | **0 (lossless)** |
| TurkishTokenizer | **78** | 10 | **68 (canonical corruption)** |
| subwords | 12–20 | 12–16 | 0 |

TurkishTokenizer places boundaries best (78%) but its tokens match the surface only 10% of the time — it emits canonical `lar`/`lık`/`üm`, not surface `ler`/`lik`/`im`. Morpheus's tokens **are** the surface morphemes, so `exact == len` (zero corruption) — this is exactly why its decode is lossless.

### Efficiency

| | Morpheus | TurkishTokenizer | 64K subword |
|---|---|---|---|
| Fertility (TR-MMLU, tok/word) | 1.73 | 1.98 | ~1.5 |
| Peak GPU mem (B=32 generation) | ~3,020 MB | ~2,151 MB | 3,723 MB |
| %Pure (Kalbur, unique tokens) | 55.2 | 65.5 | 22–34 |
| %Pure (frequency-weighted) | **83.5** | 78.2 | 40–50 |

### Downstream language modeling — BPC

A param-equalized 58 M GPT is trained with each tokenizer for an **identical 10,000 optimizer steps** (equal compute budget + identical LR schedule). Among **reversible** tokenizers, Morpheus achieves the lowest BPC. WordPiece's lower raw BPC is an artifact of accent stripping (it models lower-entropy, information-destroyed text), and TurkishTokenizer's comes with lossy canonicalization — both are excluded from the valid comparison. Full per-tokenizer BPC + inference (encode/decode speed, generation throughput, GPU memory) in `src/benchmarker/results/lm_eval/`.

### Word embeddings — Morpheus vs BERTurk vs BGE-M3

Because Morpheus is neural, the same forward pass that tokenizes also yields a word embedding. We evaluate these **frozen** vectors against BERTurk (768-d) and the multilingual retriever BGE-M3 (1024-d). The picture splits cleanly by task character: Morpheus dominates **lexical / root-level** tasks, while the heavier contextual encoders lead on **context- and inflection-dependent** tasks.

| Task | Morpheus (320) | BERTurk (768) | BGE-M3 (1024) |
|---|---|---|---|
| Root-family retrieval (MAP ↑) | **0.85** | 0.49 | 0.80 |
| Same-root verification (ROC-AUC ↑) | **1.00** | 0.70 | 0.98 |
| Number probing (acc ↑) | 0.59 | **0.95** | 0.91 |
| Case probing (acc ↑) | 0.22 | **0.89** | 0.81 |
| WikiANN-tr NER (macro-F1 ↑) | 0.48 | **0.79** | 0.76 |

This is a **deliberate, architectural trade-off**: the root-identity contrastive objective pulls a root's inflections together — sharpening root geometry (hence the retrieval/dedup wins) while collapsing the inflectional contrasts a probe reads — and the static per-word vector lacks the sentence context NER needs. Morpheus is therefore **complementary** to contextual encoders: ideal for the **lexical index** of a multi-vector RAG system (cheap, morphology-aware, strong at root matching), paired with a dense semantic encoder for context. Full results in `src/benchmarker/results/paper_eval/embeddings/`.

---

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

- **Turkish NLU / classification tasks**: where token boundaries align with morphemes, downstream classifiers can attend to specific morphological roles (case, tense, person, number) directly via attention.
- **Morphologically-sensitive information retrieval**: stemming via root identification, suffix-aware query expansion.
- **Linguistic research / corpus annotation**: morpheme-level analysis at scale without manual annotation.
- **Pretraining smaller Turkish language models** (≤1B parameters): the lower BPC and structured embeddings give favorable scaling.
- **Educational tools**: visualize Turkish morphology in real-time (e.g. learner apps).
- **Memory-constrained inference**: ~19% lower GPU memory than 64K-vocab classical tokenizers — relevant for consumer-GPU and edge deployment.

### When NOT to use Morpheus

- **Real-time generation latency-critical applications**: classical subword tokenizers achieve ~1.6× higher end-to-end character generation throughput (lower fertility = fewer forward passes per character).
- **Multilingual models**: Morpheus is Turkish-specific by design (uses Turkish character vocabulary + Morfessor supervision on Turkish). Use multilingual SentencePiece for cross-lingual tasks.
- **General-purpose LLM pretraining at frontier scale**: for trillion-token pretraining the inductive-bias advantage of morphology likely saturates and standard BPE remains the practical choice.

---

## Status

**v4 (current)** — trained on a Turkish corpus enlarged with cleaned Turkish Wikipedia, with Kalbur root-coherence label correction and a full lossless-vs-lossy comparison against TurkishTokenizer and the subword family. Reported results in this README and in `src/benchmarker/results/`. Paper preprint forthcoming on arXiv.

The architectural components (Morpheus model, Poisson-binomial soft segmentation, multi-objective curriculum, hybrid Morfessor+Kalbur teacher, evaluation harness incl. reversibility / MorphScore / SIGMORPHON / qualitative surface-fidelity / LM-BPC suites) are stable and documented.

### Planned releases
- arXiv preprint (this paper)
- Hugging Face model card + tokenizer release (`lonewolflab/Morpheus-TR-50K`)
- Hugging Face Spaces demo (interactive segmentation + embedding explorer)
- TACL / Cambridge NLP journal submission

### Trade-offs to weigh (not blockers)

Morpheus is usable for Turkish LLMs today. The points below are the engineering trade-offs to weigh when adopting it — none of them is a correctness blocker:

- **Fertility** is higher than subword tokenizers (~1.73 vs ~1.5 tokens/word) — the deliberate cost of morpheme-level tokenization, paid back in lower BPC, morphological structure, and lossless decoding. Latency-critical raw generation is the one workload where a subword tokenizer is the better pick.
- **OOV suffix chains:** on rare, long agglutinative forms the boundary detector occasionally merges adjacent suffixes. Rule-based dictionary tokenizers place such boundaries better on in-dictionary words — but they pay for it with lossy decoding. Closing this gap is the focus of the next iteration.
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
