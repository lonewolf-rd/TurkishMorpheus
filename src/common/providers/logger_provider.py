from pathlib import Path
from src.common.logger import AppLogger

global_logger = AppLogger(Path(__file__).resolve().parent.parent / "configs")
