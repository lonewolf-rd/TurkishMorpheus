from pathlib import Path
from src.common.config_manager import ConfigManager

_BASE = Path(__file__).resolve().parents[2]

md_config = ConfigManager(_BASE / "model_development" / "configs")
bench_config = ConfigManager(_BASE / "benchmarker" / "configs")
