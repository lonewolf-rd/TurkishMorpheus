from pathlib import Path
from box import Box
import yaml
import os


class ConfigManager:
    _instance = None
    _config_cache = None

    def __new__(cls, main_config: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_all(main_config)
        return cls._instance

    @staticmethod
    def _load_yaml(file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _merge(self, base: dict, override: dict) -> dict:
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = self._merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    def _load_all(self, main_config: str = None):
        base_dir = f"{Path(__file__).parent.parent}/configs"
        main_config = main_config or os.path.join(base_dir, "configs.yaml")

        all_cfg = self._load_yaml(main_config)
        includes = all_cfg.pop("include", [])

        for inc in includes:
            inc_path = os.path.join(base_dir, inc)
            inc_data = self._load_yaml(inc_path)
            all_cfg = self._merge(all_cfg, inc_data)

        self._config_cache = Box(all_cfg, default_box=True)

    @property
    def cfg(self) -> Box:
        return self._config_cache
