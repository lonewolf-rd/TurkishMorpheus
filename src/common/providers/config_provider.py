from pathlib import Path
from src.common.config_manager import ConfigManager

config_provider = ConfigManager(Path(__file__).resolve().parent.parent / "configs")
