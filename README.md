# Turkish Tokenizer Benchmark Suite

A research-grade benchmarking framework for evaluating subword tokenization methods on Turkish text, focusing on their impact on downstream language model pretraining efficiency and morphological structure preservation.

This project compares multiple tokenization strategies under a unified and reproducible evaluation pipeline, including:

- BPE (Byte Pair Encoding)
- WordPiece
- Unigram Language Model
- Byte-level BPE
- Morphology-aware baseline (Morfessor)
- Character-CNN hybrid encoders

---

## Motivation

Tokenization plays a critical role in language model performance, especially for morphologically rich languages like Turkish.
Standard subword tokenizers often fail to capture morphological structure while optimizing for compression or frequency statistics.
This project investigates:
- How different tokenizers handle Turkish morphology
- Trade-offs between compression, fertility, and linguistic structure
- Impact on representation efficiency for LLM pretraining

---

## Evaluation Metrics
The framework evaluates tokenizers using a unified metric suite:
- **Fertility**: Average tokens per word
- **Compression Ratio**: Sequence efficiency
- **Vocabulary Coverage / OOV Rate**
- **Subword Entropy**: Token distribution balance
- **Morphological Alignment Score** (approx. via Morfessor)
- **Encoding Speed (ms)**
- **Qualitative segmentation analysis**

---

## Pipeline Overview

The system is modular and reproducible:

1. **Corpus Preprocessing**
   - Text normalization
   - Noise and URL removal
   - Controlled filtering

2. **Dataset Construction**
   - Deterministic train/test split
   - Fixed evaluation sets

3. **Tokenizer Training**
   - SentencePiece (BPE / Unigram)
   - HuggingFace Tokenizers (WordPiece)
   - Morfessor baseline

4. **Evaluation**
   - Quantitative benchmarking
   - Qualitative segmentation analysis
   - Cross-tokenizer comparison
---
