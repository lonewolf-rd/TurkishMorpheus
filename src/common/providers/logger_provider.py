from pathlib import Path
from src.common.logger import AppLogger

_BASE = Path(__file__).resolve().parents[2]

md_logger = AppLogger(_BASE / "model_development" / "configs")
bench_logger = AppLogger(_BASE / "benchmarker" / "configs")
