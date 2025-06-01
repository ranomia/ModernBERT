"""
モデル性能評価用のメトリクス計算ユーティリティ
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 日本語フォント対応
try:
    import japanize_matplotlib

    print("✅ japanize_matplotlib を読み込みました。日本語表示が有効です。")
except ImportError:
    print(
        "⚠️  japanize_matplotlib が見つかりません。日本語が文字化けする可能性があります。"
    )
    print("   pip install japanize-matplotlib でインストールしてください。")


class MetricsCalculator:
    """メトリクス計算クラス"""

    def __init__(self):
        self.results = {}

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """
        基本的なメトリクスを計算

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            model_name: モデル名

        Returns:
            メトリクス辞書
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        self.results[model_name] = metrics
        return metrics

    def calculate_performance_metrics(
        self,
        inference_times: List[float],
        memory_usage: List[float],
        batch_sizes: List[int],
        model_name: str,
    ) -> Dict[str, float]:
        """
        性能メトリクスを計算

        Args:
            inference_times: 推論時間のリスト
            memory_usage: メモリ使用量のリスト
            batch_sizes: バッチサイズのリスト
            model_name: モデル名

        Returns:
            性能メトリクス辞書
        """
        total_samples = sum(batch_sizes)
        total_time = sum(inference_times)

        performance_metrics = {
            "avg_inference_time_per_batch": np.mean(inference_times),
            "total_inference_time": total_time,
            "throughput_samples_per_second": total_samples / total_time
            if total_time > 0
            else 0,
            "avg_memory_usage_mb": np.mean(memory_usage),
            "max_memory_usage_mb": np.max(memory_usage),
            "total_samples": total_samples,
            "total_batches": len(batch_sizes),
        }

        # 既存の結果に追加
        if model_name in self.results:
            self.results[model_name].update(performance_metrics)
        else:
            self.results[model_name] = performance_metrics

        return performance_metrics

    def print_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> None:
        """分類レポートを出力"""
        print(f"\n=== {model_name} Classification Report ===")
        print(classification_report(y_true, y_pred, zero_division=0))

    def print_comparison(self) -> None:
        """モデル比較結果を出力"""
        if len(self.results) < 2:
            print("比較するには少なくとも2つのモデルの結果が必要です。")
            return

        print("\n=== モデル比較結果 ===")

        # DataFrameを作成
        df = pd.DataFrame(self.results).T

        # 主要メトリクスを出力
        key_metrics = [
            "accuracy",
            "f1_macro",
            "avg_inference_time_per_batch",
            "throughput_samples_per_second",
            "avg_memory_usage_mb",
        ]

        print("\n主要メトリクス:")
        for metric in key_metrics:
            if metric in df.columns:
                print(f"\n{metric}:")
                for model_name in df.index:
                    value = df.loc[model_name, metric]
                    if isinstance(value, float):
                        if metric == "accuracy" or metric.startswith("f1"):
                            print(f"  {model_name}: {value:.4f}")
                        elif "time" in metric:
                            print(f"  {model_name}: {value:.4f}秒")
                        elif "throughput" in metric:
                            print(f"  {model_name}: {value:.2f}サンプル/秒")
                        elif "memory" in metric:
                            print(f"  {model_name}: {value:.2f}MB")
                        else:
                            print(f"  {model_name}: {value:.4f}")
                    else:
                        print(f"  {model_name}: {value}")

        # 勝敗判定
        print("\n=== 勝敗判定 ===")
        model_names = list(self.results.keys())
        if len(model_names) == 2:
            model1, model2 = model_names

            # 精度比較
            if "accuracy" in df.columns:
                acc1 = df.loc[model1, "accuracy"]
                acc2 = df.loc[model2, "accuracy"]
                if acc1 > acc2:
                    print(f"精度: {model1} が勝利 ({acc1:.4f} vs {acc2:.4f})")
                elif acc2 > acc1:
                    print(f"精度: {model2} が勝利 ({acc2:.4f} vs {acc1:.4f})")
                else:
                    print(f"精度: 引き分け ({acc1:.4f})")

            # 速度比較
            if "throughput_samples_per_second" in df.columns:
                throughput1 = df.loc[model1, "throughput_samples_per_second"]
                throughput2 = df.loc[model2, "throughput_samples_per_second"]
                if throughput1 > throughput2:
                    print(
                        f"速度: {model1} が勝利 ({throughput1:.2f} vs {throughput2:.2f} サンプル/秒)"
                    )
                elif throughput2 > throughput1:
                    print(
                        f"速度: {model2} が勝利 ({throughput2:.2f} vs {throughput1:.2f} サンプル/秒)"
                    )
                else:
                    print(f"速度: 引き分け ({throughput1:.2f} サンプル/秒)")

            # メモリ効率比較（少ない方が良い）
            if "avg_memory_usage_mb" in df.columns:
                memory1 = df.loc[model1, "avg_memory_usage_mb"]
                memory2 = df.loc[model2, "avg_memory_usage_mb"]
                if memory1 < memory2:
                    print(
                        f"メモリ効率: {model1} が勝利 ({memory1:.2f} vs {memory2:.2f} MB)"
                    )
                elif memory2 < memory1:
                    print(
                        f"メモリ効率: {model2} が勝利 ({memory2:.2f} vs {memory1:.2f} MB)"
                    )
                else:
                    print(f"メモリ効率: 引き分け ({memory1:.2f} MB)")

    def save_results(self, filename: str = "results.json") -> None:
        """結果をJSONファイルに保存"""
        import json

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"結果を {filename} に保存しました。")

    def create_comparison_plots(self, save_path: str = "comparison_plots.png") -> None:
        """比較用のプロットを作成"""
        if len(self.results) < 2:
            print("比較プロットを作成するには少なくとも2つのモデルの結果が必要です。")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("モデル性能比較", fontsize=16)

        df = pd.DataFrame(self.results).T

        # 精度比較
        if "accuracy" in df.columns and "f1_macro" in df.columns:
            ax1 = axes[0, 0]
            metrics_to_plot = ["accuracy", "f1_macro"]
            df[metrics_to_plot].plot(kind="bar", ax=ax1)
            ax1.set_title("精度メトリクス")
            ax1.set_ylabel("スコア")
            ax1.legend()
            ax1.set_xticklabels(df.index, rotation=45)

        # 推論時間比較
        if "avg_inference_time_per_batch" in df.columns:
            ax2 = axes[0, 1]
            df["avg_inference_time_per_batch"].plot(kind="bar", ax=ax2)
            ax2.set_title("平均推論時間（バッチあたり）")
            ax2.set_ylabel("時間（秒）")
            ax2.set_xticklabels(df.index, rotation=45)

        # スループット比較
        if "throughput_samples_per_second" in df.columns:
            ax3 = axes[1, 0]
            df["throughput_samples_per_second"].plot(kind="bar", ax=ax3)
            ax3.set_title("スループット")
            ax3.set_ylabel("サンプル/秒")
            ax3.set_xticklabels(df.index, rotation=45)

        # メモリ使用量比較
        if "avg_memory_usage_mb" in df.columns:
            ax4 = axes[1, 1]
            df["avg_memory_usage_mb"].plot(kind="bar", ax=ax4)
            ax4.set_title("平均メモリ使用量")
            ax4.set_ylabel("メモリ（MB）")
            ax4.set_xticklabels(df.index, rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"比較プロットを {save_path} に保存しました。")
        plt.show()


if __name__ == "__main__":
    # テスト用コード
    calculator = MetricsCalculator()

    # サンプルデータ
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred1 = np.array([0, 1, 2, 0, 1, 1, 0, 1])  # Model 1
    y_pred2 = np.array([0, 1, 2, 1, 1, 2, 0, 2])  # Model 2

    # メトリクス計算
    metrics1 = calculator.calculate_metrics(y_true, y_pred1, "Model 1")
    metrics2 = calculator.calculate_metrics(y_true, y_pred2, "Model 2")

    # 性能メトリクス
    calculator.calculate_performance_metrics(
        [0.1, 0.12, 0.11], [100, 105, 102], [8, 8, 8], "Model 1"
    )
    calculator.calculate_performance_metrics(
        [0.08, 0.09, 0.085], [80, 85, 82], [8, 8, 8], "Model 2"
    )

    # 結果出力
    calculator.print_comparison()
