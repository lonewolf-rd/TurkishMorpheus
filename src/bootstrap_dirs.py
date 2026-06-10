from pathlib import Path


BASE = Path(__file__).resolve().parent.parent


INPUT_DIRS = [
    "data/corpus_collector_tr",
    "data/sigmorphon_tr",
    "data/trmmlu",
    "data/tr_lexicon",
    "data/morphscore",
]

ARTIFACT_DIRS = [
    "src/model_development/artifacts/datasets/raw",
    "src/model_development/artifacts/datasets/splits",
    "src/model_development/artifacts/tokenizers/classical",
    "src/model_development/artifacts/tokenizers/morpheus_50k",
    "src/model_development/artifacts/checkpoints",
    "src/model_development/artifacts/lm_eval_cache",
]

RESULT_DIRS = [
    "src/benchmarker/results/lm_eval/full",
    "src/benchmarker/results/lm_eval/inference",
    "src/benchmarker/results/paper_eval",
    "src/benchmarker/results/paper_eval/sigmorphon",
    "src/benchmarker/results/paper_eval/trmmlu",
    "src/benchmarker/results/paper_eval/morphscore",
]

ALL_DIRS = INPUT_DIRS + ARTIFACT_DIRS + RESULT_DIRS


def main():
    for rel in ALL_DIRS:
        path = BASE / rel
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {rel}")
    print(f"\nCreated/verified {len(ALL_DIRS)} directories under {BASE}")


if __name__ == "__main__":
    main()
