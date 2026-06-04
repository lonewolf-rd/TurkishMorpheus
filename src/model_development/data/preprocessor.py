from src.common.providers.config_provider import md_config as config_provider
from src.common.providers.logger_provider import md_logger as global_logger
from pathlib import Path
import shutil


class DataPreprocessor:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    _RAW_FILENAME = "corpus.txt"

    def __init__(self):
        self.configs = config_provider.cfg

        self.raw_dir = self._PROJECT_ROOT / "src/model_development/artifacts/datasets/raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        local_path_str = self.configs.dataset.get("local_corpus_path")
        if not local_path_str:
            raise ValueError(
                "[DataPreprocessor] dataset.local_corpus_path is not set in model_development/configs/main.yaml"
            )
        self.local_corpus_path = (self._PROJECT_ROOT / local_path_str).resolve()
        self.raw_corpus_path = self.raw_dir / self._RAW_FILENAME

    def _validate_source(self) -> None:
        if not self.local_corpus_path.exists():
            raise FileNotFoundError(
                f"[DataPreprocessor] Local corpus not found at {self.local_corpus_path}"
            )
        size_mb = self.local_corpus_path.stat().st_size / (1024 * 1024)
        global_logger.info(
            f"[DataPreprocessor] Source corpus: {self.local_corpus_path} ({size_mb:.1f} MB)"
        )

    def _ingest(self) -> Path:
        if self.raw_corpus_path.exists():
            src_size = self.local_corpus_path.stat().st_size
            dst_size = self.raw_corpus_path.stat().st_size
            if src_size == dst_size:
                global_logger.info(
                    f"[DataPreprocessor] Raw corpus already at {self.raw_corpus_path} (size matches) — skipping copy"
                )
                return self.raw_corpus_path
            global_logger.info(
                f"[DataPreprocessor] Raw corpus exists but size differs (src={src_size}, dst={dst_size}) — overwriting"
            )

        global_logger.info(
            f"[DataPreprocessor] Copying {self.local_corpus_path.name} -> {self.raw_corpus_path}"
        )
        shutil.copy2(self.local_corpus_path, self.raw_corpus_path)
        global_logger.info(
            f"[DataPreprocessor] Copy complete ({self.raw_corpus_path.stat().st_size / (1024*1024):.1f} MB)"
        )
        return self.raw_corpus_path

    def process_pipeline(self) -> Path:
        global_logger.info("[DataPreprocessor](process_pipeline) Ingesting local Turkish corpus.")
        self._validate_source()
        return self._ingest()


if __name__ == "__main__":
    DataPreprocessor().process_pipeline()
