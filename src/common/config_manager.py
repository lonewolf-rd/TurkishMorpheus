from pathlib import Path
from box import Box
import yaml
import os


class ConfigManager:
    _instances = {}

    def __new__(cls, configs_dir):
        configs_dir = Path(configs_dir)
        key = str(configs_dir.resolve())
        if key not in cls._instances:
            instance = super(ConfigManager, cls).__new__(cls)
            instance._load_all(configs_dir)
            cls._instances[key] = instance
        return cls._instances[key]

    @staticmethod
    def _load_yaml(file_path) -> dict:
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

    def _load_all(self, configs_dir: Path):
        main_config = configs_dir / "configs.yaml"

        all_cfg = self._load_yaml(main_config)
        includes = all_cfg.pop("include", [])

        for inc in includes:
            inc_path = configs_dir / inc
            inc_data = self._load_yaml(inc_path)
            all_cfg = self._merge(all_cfg, inc_data)

        self._config_cache = Box(all_cfg, default_box=True)

    @property
    def cfg(self) -> Box:
        return self._config_cache
