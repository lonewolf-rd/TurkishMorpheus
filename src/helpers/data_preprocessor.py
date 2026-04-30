from src.utils.providers.config_provider import config_provider
from src.utils.logger import AppLogger
from pathlib import Path
from typing import Optional, List
from datasets import load_dataset
import re, os


class DataPreprocessor:
    def __init__(self):
        self.logger = AppLogger()
        self.configs = config_provider.cfg

        self.base_path = Path(__file__).parent.parent
        self.dataset_dir = self.base_path / "dataset"
        self.raw_dir = self.dataset_dir / "raw"

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = self.configs.huggingface.access_token
        os.environ["HF_TOKEN"] = self.configs.huggingface.access_token

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fetch_wikipedia_tr(self, limit_mb: Optional[int] = None) -> Path:
        self.logger.info(f"[DataPreprocessor](fetch_wikipedia_tr) Starting fetch process for Wikipedia TR.")
        output_path = self.raw_dir / "wiki_tr_raw.txt"

        try:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.tr",
                split="train",
                streaming=True
            )

            with open(output_path, "w", encoding="utf-8") as f:
                current_size = 0
                for item in dataset:
                    cleaned_text = self._clean_text(text=item['text'])
                    if cleaned_text:
                        f.write(cleaned_text + "\n")
                        current_size += len(cleaned_text.encode('utf-8'))

                    if limit_mb and current_size >= (limit_mb * 1024 * 1024):
                        break

            self.logger.info(f"[DataPreprocessor](fetch_wikipedia_tr) Successfully saved raw data to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"[DataPreprocessor](fetch_wikipedia_tr) Error occurred during fetch: {str(e)}")
            raise

    def fetch_oscar_tr(self, limit_mb: Optional[int] = None) -> Path:
        self.logger.info(f"[DataPreprocessor](fetch_oscar_tr) Starting fetch process for OSCAR TR.")
        output_path = self.raw_dir / "oscar_tr_raw.txt"

        try:
            dataset = load_dataset(
                "oscar-corpus/OSCAR-2301",
                "tr",
                split="train",
                streaming=True,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                current_size = 0
                for item in dataset:
                    cleaned_text = self._clean_text(item['text'])
                    if cleaned_text:
                        f.write(cleaned_text + "\n")
                        current_size += len(cleaned_text.encode('utf-8'))

                    if limit_mb and current_size >= (limit_mb * 1024 * 1024):
                        break

            self.logger.info(f"[DataPreprocessor](fetch_oscar_tr) Successfully saved raw data to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"[DataPreprocessor](fetch_oscar_tr) Error occurred during fetch: {str(e)}")
            raise

    def merge_and_finalize(self, file_paths: List[Path], output_filename: str = "final_corpus.txt") -> Path:
        final_path = self.dataset_dir / output_filename
        self.logger.info(f"[DataPreprocessor](merge_and_finalize) Merging {len(file_paths)} files into {final_path}")

        try:
            with open(final_path, "w", encoding="utf-8") as outfile:
                for file_path in file_paths:
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as infile:
                            for line in infile:
                                outfile.write(line)
                        self.logger.info(f"[DataPreprocessor](merge_and_finalize) Appended {file_path.name}")

            self.logger.info(f"[DataPreprocessor](merge_and_finalize) Final corpus is ready at {final_path}")
            return final_path
        except Exception as e:
            self.logger.error(f"[DataPreprocessor](merge_and_finalize) Error during merging: {str(e)}")
            raise

    def process_pipeline(self):
        self.logger.info(f"[DataPreprocessor](process_pipeline) Initializing data pipeline.")
        limit = self.configs.dataset.get("limit_mb", 100)

        paths = []
        try:
            if self.configs.dataset.get("fetch_wiki", True):
                paths.append(self.fetch_wikipedia_tr(limit_mb=limit))

            if self.configs.dataset.get("fetch_oscar", False):
                paths.append(self.fetch_oscar_tr(limit_mb=limit))

            if paths:
                self.merge_and_finalize(paths)
            else:
                self.logger.warning("[DataPreprocessor](process_pipeline) No data sources selected in configuration.")

        except Exception as e:
            self.logger.error(f"[DataPreprocessor](process_pipeline) Pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.process_pipeline()
