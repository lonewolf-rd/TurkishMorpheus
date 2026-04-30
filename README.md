# Turkish Tokenizer Benchmark Suite

Current Large Language Models (LLMs) primarily rely on subword tokenization algorithms (e.g., BPE, WordPiece, Unigram) trained on massive multilingual corpora. While these methods demonstrate remarkable generalization across high-resource languages, they exhibit a significant structural deficit when applied to agglutinative languages like Turkish.
The Morphological Gap

In Turkish, semantic information is densely packed into suffixes (agglutination). Standard statistical tokenizers optimize for frequency and compression rather than linguistic or semantic boundaries. This leads to:

* Semantic Fragmentation: Suffixes that carry critical grammatical and semantic weight (tense, person, negation) are often fragmented or merged inconsistently, leading to a loss of morphological transparency.
* Model Decay in Small-Scale LLMs: While trilinear-parameter models can compensate for poor tokenization via sheer scale, smaller or specialized models often show "linguistic drift" or performance degradation in Turkish as they fail to capture the underlying generative rules of the language.
* Data Inefficiency: The lack of a "morpheme-aware" tokenizer forces models to learn thousands of redundant surface forms instead of a finite set of roots and functional suffixes.

## Research Objective

This project is a comparative study aimed at bridging the gap between statistical compression and semantic-morphological representation. The research investigates whether moving away from traditional discrete tokenization toward pattern-aware architectures can preserve Turkish linguistic structure more effectively.
Core Research Goals:

 * Quantitative Benchmarking: Systematically comparing standard subword tokenizers against Character-level CNN (CharCNN) encoders that treat words as continuous signals of semantic patterns rather than discrete chunks.
 * Semantic Pattern Acquisition: Evaluating the ability of different encoders to maintain the relationship between a root and its various functional suffixes in the latent space.
 * Metric-Driven Evaluation: Moving beyond "Fertility" and "Compression" to include Morphological Alignment Scores (MAS) and Subword Entropy, providing a holistic view of how well a tokenizer "understands" the language it processes.

## Methodology

----
| Method | Learning Paradigm | Unit Level | Research Significance | Turkish Specific Role |
| :--- | :--- | :--- | :--- | :--- |
| **BPE** | Frequency-based Merging | Subword | Standard baseline for most LLMs (GPT-series). | Evaluates greedy statistical merging efficiency. |
| **Byte-BPE** | Byte-level Frequency | Byte | Robust against OOV; standard for Llama/Gemma. | Tests resilience against noisy/special Turkish characters. |
| **Unigram** | Probabilistic Sampling | Subword | Flexible, entropy-based subword selection. | Benchmarks probabilistic vs. greedy segmentation. |
| **Morfessor** | MDL (Linguistic-based) | Morpheme | **Ground Truth Baseline:** Unsupervised morphology. | Provides the "gold standard" for suffix/root boundaries. |
| **Char-CNN** | Feature Extraction | Character | **Hybrid/Token-free:** Captures semantic patterns. | Investigates non-discrete, pattern-aware representation. |
----

## Evaluation

To validate our objectives, we utilize a multidimensional metric suite:
 * Morphological Integrity: Measured via alignment with Morfessor-generated ground truth.
 * Sequence Efficiency: Analysis of the "Fertility Rate" and its impact on the effective context window.
 * Vocab Utilization: Entropy-based analysis of how effectively the model utilizes its allocated vocabulary.
 * Semantic Robustness: Qualitative analysis of how the encoder handles long-range dependencies within complex Turkish words (e.g., "Muvaffakiyetsizleştiriciler...").

```bash
@research_note{lonewolfrd2026,
  title={Morphological Fragmentation in Turkish: A Comparative Study of Subword vs. Semantic Encoders},
  author={Tolga, Şakar},
  institution={lonewolfrd},
  year={2026}
}
```