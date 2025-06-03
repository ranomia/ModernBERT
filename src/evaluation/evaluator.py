"""
モデル評価を実行する評価器クラス
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import JCommonsenseQALoader
from models.bert_model import BERTModel
from models.modern_bert_model import ModernBERTModel
from utils.metrics import MetricsCalculator


class ModelEvaluator:
    """モデル評価クラス"""

    def __init__(self, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.metrics_calculator = MetricsCalculator()
        print(f"Using device: {self.device}")

    def evaluate_model(self, model, data_loader, model_name: str) -> Dict[str, Any]:
        """
        単一モデルの評価を実行

        Args:
            model: 評価対象のモデル
            data_loader: データローダー
            model_name: モデル名

        Returns:
            評価結果
        """
        model.eval()
        model.to(self.device)

        all_predictions = []
        all_labels = []
        all_inference_times = []
        all_memory_usage = []
        all_batch_sizes = []

        print(f"\n{model_name} の評価を開始...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(data_loader, desc=f"Evaluating {model_name}")
            ):
                # 予測実行
                result = model.predict(batch, self.device)

                # 結果を収集
                all_predictions.extend(result["predictions"])
                all_labels.extend(batch["labels"].numpy())
                all_inference_times.append(result["inference_time"])
                all_memory_usage.append(result["memory_used"])
                all_batch_sizes.append(result["batch_size"])

        # numpy配列に変換
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # メトリクス計算
        accuracy_metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions, model_name
        )

        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            all_inference_times, all_memory_usage, all_batch_sizes, model_name
        )

        # 分類レポート出力
        self.metrics_calculator.print_classification_report(
            all_labels, all_predictions, model_name
        )

        # モデル情報取得
        model_info = model.get_model_info()

        # 結果をまとめる
        evaluation_result = {
            "model_name": model_name,
            "model_info": model_info,
            "accuracy_metrics": accuracy_metrics,
            "performance_metrics": performance_metrics,
            "predictions": all_predictions,
            "labels": all_labels,
            "total_samples": len(all_predictions),
        }

        return evaluation_result

    def compare_models(
        self,
        models_config: List[Dict[str, Any]],
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        複数モデルの比較評価を実行

        Args:
            models_config: モデル設定のリスト
                例: [{'name': 'BERT', 'class': BERTModel, 'model_name': 'cl-tohoku/bert-base-japanese-v3'}]
            batch_size: バッチサイズ
            max_length: 最大系列長

        Returns:
            比較結果
        """
        print(f"=== ModernBERT vs BERT 性能比較評価 ===")
        print(f"バッチサイズ: {batch_size}")
        print(f"最大系列長: {max_length}")
        print(f"デバイス: {self.device}")

        results = {}

        for model_config in models_config:
            model_name = model_config["name"]
            model_class = model_config["class"]
            model_args = model_config.get("args", {})

            print(f"\n--- {model_name} の評価 ---")

            # モデル初期化
            model = model_class(**model_args)

            # データローダー作成（各モデルのトークナイザーを使用）
            data_loader_instance = JCommonsenseQALoader(
                tokenizer_name=model_args.get(
                    "model_name", "cl-tohoku/bert-base-japanese-v3"
                )
            )
            data_loader = data_loader_instance.create_dataloader(
                split="validation",  # 最終評価用データセット（testデータが非公開のため）
                batch_size=batch_size,
                max_length=max_length,
                shuffle=False,
            )

            # 評価実行
            result = self.evaluate_model(model, data_loader, model_name)
            results[model_name] = result

            # メモリ解放
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 比較結果出力
        print("\n" + "=" * 80)
        self.metrics_calculator.print_comparison()

        # 結果保存
        self.save_comparison_results(results)

        return results

    def save_comparison_results(
        self, results: Dict[str, Any], output_dir: str = "results"
    ) -> None:
        """比較結果を保存"""
        os.makedirs(output_dir, exist_ok=True)

        # メトリクス結果をJSON形式で保存
        self.metrics_calculator.save_results(
            os.path.join(output_dir, "metrics_comparison.json")
        )

        # 比較プロット作成・保存
        try:
            self.metrics_calculator.create_comparison_plots(
                os.path.join(output_dir, "comparison_plots.png")
            )
        except Exception as e:
            print(f"プロット作成でエラーが発生しました: {e}")

        # 詳細結果をnumpy形式で保存
        for model_name, result in results.items():
            np.savez(
                os.path.join(output_dir, f"{model_name.lower()}_predictions.npz"),
                predictions=result["predictions"],
                labels=result["labels"],
            )

        print(f"\n結果を {output_dir} ディレクトリに保存しました。")

    def run_full_evaluation(self, batch_size: int = 8, max_length: int = 512) -> None:
        """フル評価を実行"""
        models_config = [
            {
                "name": "BERT (東北大v3)",
                "class": BERTModel,
                "args": {
                    "model_name": "cl-tohoku/bert-base-japanese-v3",
                    "num_choices": 5,
                },
            },
            {
                "name": "ModernBERT (SB-Intuitions)",
                "class": ModernBERTModel,
                "args": {
                    "model_name": "SB-Intuitions/ModernBERT-Ja-130M",
                    "num_choices": 5,
                },
            },
        ]

        # 比較評価実行
        results = self.compare_models(models_config, batch_size, max_length)

        # 最終サマリー出力
        print("\n" + "=" * 80)
        print("=== 評価完了 ===")
        print(f"評価対象データセット: JCommonsenseQA")
        print(f"評価されたモデル数: {len(results)}")

        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  - 総サンプル数: {result['total_samples']}")
            print(f"  - 正解率: {result['accuracy_metrics']['accuracy']:.4f}")
            print(f"  - F1スコア (Macro): {result['accuracy_metrics']['f1_macro']:.4f}")
            print(f"  - 総パラメータ数: {result['model_info']['total_parameters']:,}")
            print(f"  - モデルサイズ: {result['model_info']['model_size_mb']:.1f}MB")


if __name__ == "__main__":
    # 評価実行
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation(
        batch_size=4, max_length=256
    )  # メモリ使用量を抑えるために小さめの設定
