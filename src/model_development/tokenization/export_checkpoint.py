"""Export a small, inference-only Morpheus checkpoint.

    python -m src.model_development.tokenization.export_checkpoint
    python -m src.model_development.tokenization.export_checkpoint --src path/to/best.pt --dst out.pt
"""
import argparse
from pathlib import Path

from src.model_development.tokenization.morpheus_tokenizer import (
    INFERENCE_CHECKPOINT,
    TRAINING_CHECKPOINT,
    export_inference_checkpoint,
)


def main():
    checkpoints = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints"
    parser = argparse.ArgumentParser(prog="src.model_development.tokenization.export_checkpoint")
    parser.add_argument("--src", default=str(checkpoints / TRAINING_CHECKPOINT))
    parser.add_argument("--dst", default=str(checkpoints / INFERENCE_CHECKPOINT))
    args = parser.parse_args()
    out = export_inference_checkpoint(args.src, args.dst)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
