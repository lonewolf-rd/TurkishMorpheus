from threading import Lock
from pathlib import Path
import logging.config
import logging, yaml, os


class AppLogger:
    _instances = {}
    _lock = Lock()

    def __new__(cls, configs_dir):
        configs_dir = Path(configs_dir)
        key = str(configs_dir.resolve())
        with cls._lock:
            if key not in cls._instances:
                instance = super(AppLogger, cls).__new__(cls)
                instance._initialize(configs_dir)
                cls._instances[key] = instance
            return cls._instances[key]

    def _initialize(self, configs_dir: Path):
        logging_config_file = configs_dir / "logging.yaml"

        log_cfg = None
        if os.path.exists(logging_config_file):
            with open(logging_config_file, "r") as f:
                log_cfg = yaml.safe_load(f)
            logging.config.dictConfig(log_cfg["logging"])
        else:
            logging.basicConfig(level=logging.INFO)

        app_name = log_cfg["logging"]["app_name"] if log_cfg else "app_logger"
        self._logger = logging.getLogger(app_name)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def get_logger(name: str = "app_logger") -> logging.Logger:
        return logging.getLogger(name)
