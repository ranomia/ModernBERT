"""
ファインチューニング機能付きModernBERT vs BERT 性能比較メインスクリプト
"""

import argparse
import sys
import os
import torch
import japanize_matplotlib
from datetime import datetime

# モジュールインポート
from evaluation.evaluator import ModelEvaluator
from models.bert_model import BERTModel
from models.modern_bert_model import ModernBERTModel
from training.trainer import SimpleTrainer


def get_device():
    """最適なデバイスを自動選択"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA GPU検出: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple MPS GPU検出")
        return device
    else:
        device = torch.device("cpu")
        print("⚠️  GPU未検出 - CPUを使用します")
        return device


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="ModernBERT vs BERT 性能比較評価（ファインチューニング付き）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "modern_bert", "both"],
        default="both",
        help="評価するモデル",
    )

    # GPU使用時の推奨バッチサイズに変更
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="バッチサイズ（GPU使用時推奨：16-32）",
    )
    parser.add_argument("--max_length", type=int, default=128, help="最大系列長")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="使用デバイス（autoで自動選択）",
    )

    # ファインチューニング設定
    parser.add_argument(
        "--finetune", action="store_true", help="ファインチューニングを実行"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="ファインチューニングエポック数"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学習率")

    # GPU最適化オプション
    parser.add_argument(
        "--mixed_precision", action="store_true", help="混合精度学習を使用（GPU推奨）"
    )

    return parser.parse_args()


def setup_gpu_optimizations(device):
    """GPU使用時の最適化設定"""
    if device.type == "cuda":
        # cuDNN最適化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # メモリ最適化
        torch.cuda.empty_cache()

        print("✅ GPU最適化設定完了:")
        print("   - cuDNN benchmark: 有効")
        print("   - メモリキャッシュ: クリア済み")


def finetune_and_evaluate(model_config, args, device):
    """ファインチューニングと評価を実行"""
    model_name = model_config["name"]
    model_class = model_config["class"]
    model_args = model_config["args"]

    print(f"\n--- {model_name} の処理開始 ---")

    # モデル初期化
    model = model_class(**model_args)

    if args.finetune:
        print(f"🎓 {model_name} のファインチューニングを実行中...")

        # ファインチューニング実行
        trainer = SimpleTrainer(
            model,
            device,
            learning_rate=args.learning_rate,
            use_mixed_precision=args.mixed_precision,
            enable_plotting=True,  # 学習曲線プロット機能を有効化
        )
        model = trainer.quick_finetune(
            tokenizer_name=model_args.get(
                "model_name", "cl-tohoku/bert-base-japanese-v3"
            ),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_name=model_name,  # モデル名を渡す
        )
    else:
        print(
            f"⚠️  {model_name} はファインチューニングなしで評価します（性能が低い可能性があります）"
        )

    # 評価実行
    evaluator = ModelEvaluator(device=device)

    # データローダー作成
    from data.data_loader import JCommonsenseQALoader

    data_loader_instance = JCommonsenseQALoader(
        tokenizer_name=model_args.get("model_name", "cl-tohoku/bert-base-japanese-v3")
    )
    data_loader = data_loader_instance.create_dataloader(
        split="validation",  # 最終評価用データセット（testデータが非公開のためvalidationを使用）
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    result = evaluator.evaluate_model(model, data_loader, model_name)

    # メモリ解放（GPU使用時重要）
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def main():
    """メイン関数"""
    print("📚 ModernBERT vs BERT 性能比較評価システム（ファインチューニング対応）")
    print("=" * 80)

    args = parse_arguments()

    # デバイス設定
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        print(f"🔧 手動指定デバイス: {device}")

    # GPU最適化設定
    if device.type in ["cuda", "mps"]:
        setup_gpu_optimizations(device)

    print(f"\n🔧 設定:")
    print(f"  - デバイス: {device}")
    print(f"  - バッチサイズ: {args.batch_size}")
    print(f"  - 最大系列長: {args.max_length}")
    print(f"  - ファインチューニング: {'✅ 有効' if args.finetune else '❌ 無効'}")
    print(f"  - 混合精度学習: {'✅ 有効' if args.mixed_precision else '❌ 無効'}")
    if args.finetune:
        print(f"  - エポック数: {args.epochs}")
        print(f"  - 学習率: {args.learning_rate}")

    # GPU使用時の推奨設定アドバイス
    if device.type == "cuda" and not args.mixed_precision:
        print("\n💡 GPU最適化のヒント:")
        print("   --mixed_precision オプションで学習を高速化できます")

    if device.type == "cuda" and args.batch_size < 16:
        print(f"💡 GPU使用時は、バッチサイズを16以上に設定することを推奨します")

    # モデル設定
    models_config = []

    if args.model in ["bert", "both"]:
        models_config.append(
            {
                "name": "BERT (東北大v3)",
                "class": BERTModel,
                "args": {
                    "model_name": "cl-tohoku/bert-base-japanese-v3",
                    "num_choices": 5,
                },
            }
        )

    if args.model in ["modern_bert", "both"]:
        models_config.append(
            {
                "name": "ModernBERT (SB-Intuitions)",
                "class": ModernBERTModel,
                "args": {
                    "model_name": "sbintuitions/modernbert-ja-130m",
                    "num_choices": 5,
                },
            }
        )

    results = {}

    # 各モデルを順次処理
    for model_config in models_config:
        try:
            result = finetune_and_evaluate(model_config, args, device)
            results[model_config["name"]] = result
        except Exception as e:
            print(f"❌ {model_config['name']} でエラーが発生しました: {e}")
            # GPU メモリ不足の場合のアドバイス
            if "CUDA out of memory" in str(e):
                print("💡 GPU メモリ不足の解決策:")
                print("   - バッチサイズを減らす: --batch_size 8")
                print("   - 系列長を短くする: --max_length 64")
                print("   - 混合精度を使用: --mixed_precision")
            continue

    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 **最終結果サマリー**")
    print("=" * 80)

    if not results:
        print("❌ 評価結果がありません。")
        return

    print(f"📋 データセット: JCommonsenseQA (1,119サンプル)")
    print(f"🎯 理論的ランダム性能: 20.0% (5択問題)")
    print(f"⚙️  ファインチューニング: {'実行済み' if args.finetune else '未実行'}")
    print(f"🔧 実行デバイス: {device}")

    for model_name, result in results.items():
        accuracy = result["accuracy_metrics"]["accuracy"]
        f1_score = result["accuracy_metrics"]["f1_macro"]

        # 性能判定
        if accuracy > 0.6:
            performance_icon = "🏆"
            performance_text = "優秀"
        elif accuracy > 0.4:
            performance_icon = "🥈"
            performance_text = "良好"
        elif accuracy > 0.25:
            performance_icon = "🥉"
            performance_text = "普通"
        else:
            performance_icon = "⚠️"
            performance_text = "要改善"

        print(f"\n{performance_icon} **{model_name}** ({performance_text})")
        print(f"   ✅ 正解率: {accuracy:.1%}")
        print(f"   📊 F1スコア: {f1_score:.1%}")
        print(f"   🧮 パラメータ数: {result['model_info']['total_parameters']:,}")

    if not args.finetune:
        print("\n💡 **改善提案**:")
        print(
            "   性能向上のために、--finetuneオプションでファインチューニングを試してください:"
        )
        if device.type == "cuda":
            print(
                "   python main_with_training.py --finetune --epochs 5 --batch_size 16 --mixed_precision"
            )
        else:
            print(
                "   python main_with_training.py --finetune --epochs 5 --batch_size 4"
            )
    else:
        print("\n📈 **学習曲線について**:")
        print("   - 個別モデルの学習曲線: 各モデルのファインチューニング後に表示")
        print("   - 学習履歴データ: results/training_history_*.json")
        print("   - 学習曲線画像: results/learning_curves_*.png")

    # 最終メモリクリーンアップ
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
