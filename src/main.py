"""
ModernBERT vs BERT 性能比較メインスクリプト
"""

import argparse
import sys
import os
import torch
from datetime import datetime

# モジュールインポート
from evaluation.evaluator import ModelEvaluator
from models.bert_model import BERTModel
from models.modern_bert_model import ModernBERTModel


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="ModernBERT vs BERT 性能比較評価",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "modern_bert", "both"],
        default="both",
        help="評価するモデル (bert: BERT のみ, modern_bert: ModernBERT のみ, both: 両方)",
    )

    parser.add_argument("--batch_size", type=int, default=8, help="バッチサイズ")

    parser.add_argument("--max_length", type=int, default=512, help="最大系列長")

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用デバイス (cuda, cpu, または None で自動選択)",
    )

    parser.add_argument(
        "--output_dir", type=str, default="results", help="結果出力ディレクトリ"
    )

    parser.add_argument(
        "--bert_model",
        type=str,
        default="cl-tohoku/bert-base-japanese-v3",
        help="使用するBERTモデル名",
    )

    parser.add_argument(
        "--modern_bert_model",
        type=str,
        default="sbintuitions/modernbert-ja-130m",
        help="使用するModernBERTモデル名",
    )

    return parser.parse_args()


def print_system_info():
    """システム情報を出力"""
    print("=== システム情報 ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()


def setup_models_config(args):
    """モデル設定を準備"""
    models_config = []

    if args.model in ["bert", "both"]:
        models_config.append(
            {
                "name": "BERT (東北大v3)",
                "class": BERTModel,
                "args": {"model_name": args.bert_model, "num_choices": 5},
            }
        )

    if args.model in ["modern_bert", "both"]:
        models_config.append(
            {
                "name": "ModernBERT (SB-Intuitions)",
                "class": ModernBERTModel,
                "args": {"model_name": args.modern_bert_model, "num_choices": 5},
            }
        )

    return models_config


def main():
    """メイン関数"""
    print("ModernBERT vs BERT 性能比較評価システム")
    print("=" * 60)
    print(f"実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 引数解析
    args = parse_arguments()

    # システム情報出力
    print_system_info()

    # 設定情報出力
    print("=== 実行設定 ===")
    print(f"評価モデル: {args.model}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"最大系列長: {args.max_length}")
    print(f"デバイス: {args.device if args.device else '自動選択'}")
    print(f"結果出力先: {args.output_dir}")
    print(f"BERTモデル: {args.bert_model}")
    print(f"ModernBERTモデル: {args.modern_bert_model}")
    print()

    try:
        # 評価器初期化
        evaluator = ModelEvaluator(device=args.device)

        # モデル設定準備
        models_config = setup_models_config(args)

        if not models_config:
            print("エラー: 評価するモデルが選択されていません。")
            return

        # 評価実行
        print("=== 評価開始 ===")
        results = evaluator.compare_models(
            models_config=models_config,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # 結果保存
        if args.output_dir != "results":
            evaluator.save_comparison_results(results, args.output_dir)

        # 最終サマリー
        print("\n" + "=" * 80)
        print("=== 評価完了サマリー ===")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"評価対象: JCommonsenseQA データセット")
        print(f"評価モデル数: {len(results)}")

        for model_name, result in results.items():
            print(f"\n📊 {model_name}:")
            print(f"  ✅ 正解率: {result['accuracy_metrics']['accuracy']:.4f}")
            print(f"  📈 F1スコア: {result['accuracy_metrics']['f1_macro']:.4f}")
            print(
                f"  ⚡ スループット: {result['performance_metrics']['throughput_samples_per_second']:.2f} サンプル/秒"
            )
            print(
                f"  💾 平均メモリ使用量: {result['performance_metrics']['avg_memory_usage_mb']:.1f} MB"
            )
            print(f"  🔢 パラメータ数: {result['model_info']['total_parameters']:,}")
            print(f"  📦 モデルサイズ: {result['model_info']['model_size_mb']:.1f} MB")

        print(f"\n📁 詳細結果は {args.output_dir} ディレクトリに保存されました。")
        print("\n🎉 評価が正常に完了しました！")

    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによって評価が中断されました。")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        print("詳細なエラー情報については、トレースバックを確認してください。")
        raise


if __name__ == "__main__":
    main()
