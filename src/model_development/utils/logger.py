from threading import Lock
from pathlib import Path
import logging.config
import logging, yaml, os


class AppLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppLogger, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        logging_config_file = (
            Path(__file__).resolve().parent.parent / "configs/logging.yaml"
        )

        if os.path.exists(logging_config_file):
            with open(logging_config_file, "r") as f:
                log_cfg = yaml.safe_load(f)
            logging.config.dictConfig(log_cfg["logging"])

        else:
            logging.basicConfig(level=logging.INFO)

        self._logger = logging.getLogger(log_cfg["logging"]["app_name"])

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
