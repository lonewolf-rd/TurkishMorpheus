import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from pathlib import Path
from src.common.providers.logger_provider import bench_logger as global_logger


class ResultVisualizer:
    def __init__(self, output_dir: str = "results"):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['figure.dpi'] = 150

    def save_summary_table_image(self, df: pd.DataFrame, filename: str = "benchmark_summary.png"):
        cols = ["name", "fertility", "compression", "morfem_alignment", "subword_entropy", "weighted_score"]
        temp_df = df[cols].sort_values("weighted_score", ascending=False).copy()

        fig, ax = plt.subplots(figsize=(14, len(temp_df) * 0.5 + 2))
        ax.axis('off')

        table = ax.table(
            cellText=temp_df.values,
            colLabels=[c.replace("_", " ").title() for c in temp_df.columns],
            cellLoc='center',
            loc='center',
            colColours=["#2c3e50"] * len(temp_df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.5)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
            if row > 0 and col == 0:
                cell.set_text_props(ha='left')

        plt.title("Turkish Tokenizer Benchmark - Global Leaderboard", fontsize=16, pad=20, fontweight="bold")
        plt.savefig(self.output_path / filename, bbox_inches="tight")
        plt.close()
        global_logger.info(f"[Visualizer] Summary table image saved to {filename}")

    def calculate_weighted_winner(self, df: pd.DataFrame) -> pd.DataFrame:
        df_score = df.copy()
        for col in ["fertility", "oov_rate", "encode_ms"]:
            mn, mx = df_score[col].min(), df_score[col].max()
            if mx > mn:
                df_score[f"{col}_norm"] = 1 - (df_score[col] - mn) / (mx - mn)
            else:
                df_score[f"{col}_norm"] = 1.0

        for col in ["compression", "vocab_coverage", "subword_entropy", "morfem_alignment"]:
            mn, mx = df_score[col].min(), df_score[col].max()
            if mx > mn:
                df_score[f"{col}_norm"] = (df_score[col] - mn) / (mx - mn)
            else:
                df_score[f"{col}_norm"] = 1.0

        weights = {
            "fertility_norm": 0.35,
            "compression_norm": 0.20,
            "subword_entropy_norm": 0.15,
            "vocab_coverage_norm": 0.15,
            "morfem_alignment_norm": 0.10,
            "oov_rate_norm": 0.03,
            "encode_ms_norm": 0.02,
        }

        df_score["weighted_score"] = sum(
            df_score[f"{col}"] * weight for col, weight in weights.items()
        )

        return df_score.sort_values("weighted_score", ascending=False)

    def plot_metric_distribution(self, df: pd.DataFrame, filename: str = "metrics_grid.png"):
        df32 = df[df["name"].str.contains("32K|32000")].copy()
        if df32.empty: return

        metrics = [
            ("fertility", "Fertility (Lower is better)", False),
            ("compression", "Compression (Higher is better)", True),
            ("morfem_alignment", "Morfem Alignment % (Higher is better)", True),
            ("subword_entropy", "Entropy (Higher is better)", True),
            ("oov_rate", "OOV % (Lower is better)", False),
            ("encode_ms", "Speed ms (Lower is better)", False)
        ]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Detailed Metric Analysis (32K Vocab Size)", fontsize=20, fontweight="bold", y=0.95)

        for ax, (metric, title, higher_better) in zip(axes.flat, metrics):
            data = df32.sort_values(metric, ascending=higher_better)
            colors = ["#27ae60" if i == (len(data) - 1) else "#3498db" for i in range(len(data))]

            bars = ax.barh(data["name"], data[metric], color=colors, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight="bold")

            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f' {width:.3f}', va='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_path / filename)
        plt.close()
        global_logger.info(f"[Visualizer] Metrics grid saved to {filename}")

    def plot_learning_curves(self, df: pd.DataFrame, filename: str = "fertility_curves.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        methods = df["name"].apply(lambda x: x.split("-")[0]).unique()

        for method in methods:
            sub = df[df["name"].str.startswith(method)].sort_values("vocab_size")
            if sub.empty: continue

            ax1.plot(sub["vocab_size"], sub["fertility"], marker='o', label=method, linewidth=2.5)
            ax2.plot(sub["vocab_size"], sub["morfem_alignment"], marker='s', label=method, linestyle='--')

        ax1.set_title("Fertility Curve (Efficiency)", fontweight="bold")
        ax1.set_ylabel("Tokens per Word")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x) // 1000}K"))
        ax1.legend()

        ax2.set_title("Morfem Alignment Curve (Linguistic Quality)", fontweight="bold")
        ax2.set_ylabel("Alignment %")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x) // 1000}K"))
        ax2.legend()

        plt.savefig(self.output_path / filename)
        plt.close()
        global_logger.info(f"[Visualizer] Learning curves saved to {filename}")

    def run_all(self, df: pd.DataFrame):
        df.to_csv(self.output_path / "raw_results.csv", index=False)

        self.save_summary_table_image(df)
        self.plot_metric_distribution(df)
        self.plot_learning_curves(df)

        global_logger.info(f"Visualization complete. Artifacts are in '{self.output_path}/'")