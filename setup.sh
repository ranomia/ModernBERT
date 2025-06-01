#!/bin/bash

# ModernBERT vs BERT 性能検証環境セットアップスクリプト

echo "=========================================="
echo "ModernBERT vs BERT 性能検証環境セットアップ"
echo "=========================================="

# Python3がインストールされているかチェック
if ! command -v python3 &> /dev/null; then
    echo "❌ エラー: Python3がインストールされていません。"
    echo "   Python3.8以上をインストールしてから再実行してください。"
    exit 1
fi

# Pythonのバージョンをチェック
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "✅ Python $python_version を検出しました。"

# 仮想環境の作成
echo ""
echo "🔧 Python仮想環境を作成中..."
if [ -d ".venv" ]; then
    echo "⚠️  既存の.venvディレクトリが見つかりました。削除しますか？ [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf .venv
        echo "✅ 既存の仮想環境を削除しました。"
    else
        echo "❌ セットアップを中止しました。"
        exit 1
    fi
fi

python3 -m venv .venv
if [ $? -eq 0 ]; then
    echo "✅ 仮想環境を作成しました。"
else
    echo "❌ エラー: 仮想環境の作成に失敗しました。"
    exit 1
fi

# 仮想環境をアクティベート
echo ""
echo "🔧 仮想環境をアクティベート中..."
source .venv/bin/activate
if [ $? -eq 0 ]; then
    echo "✅ 仮想環境をアクティベートしました。"
else
    echo "❌ エラー: 仮想環境のアクティベートに失敗しました。"
    exit 1
fi

# pipのアップグレード
echo ""
echo "🔧 pipをアップグレード中..."
pip install --upgrade pip
if [ $? -eq 0 ]; then
    echo "✅ pipをアップグレードしました。"
else
    echo "❌ エラー: pipのアップグレードに失敗しました。"
    exit 1
fi

# 依存関係のインストール
echo ""
echo "🔧 依存関係をインストール中..."
echo "   これには数分かかる場合があります..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ 依存関係のインストールが完了しました。"
else
    echo "❌ エラー: 依存関係のインストールに失敗しました。"
    exit 1
fi

# ディレクトリ構造の確認
echo ""
echo "🔧 プロジェクト構造を確認中..."
if [ -d "src" ] && [ -f "src/main.py" ]; then
    echo "✅ プロジェクト構造が正常です。"
else
    echo "❌ エラー: プロジェクト構造に問題があります。"
    exit 1
fi

# セットアップ完了
echo ""
echo "=========================================="
echo "🎉 セットアップが完了しました！"
echo "=========================================="
echo ""
echo "次の手順で評価を実行できます："
echo ""
echo "1. 仮想環境をアクティベート："
echo "   source .venv/bin/activate"
echo ""
echo "2. srcディレクトリに移動："
echo "   cd src"
echo ""
echo "3. 評価を実行："
echo "   python main.py"
echo ""
echo "または、コマンドオプションを指定："
echo "   python main.py --model both --batch_size 4 --max_length 256"
echo ""
echo "利用可能なオプション："
echo "   --model: bert, modern_bert, both (デフォルト: both)"
echo "   --batch_size: バッチサイズ (デフォルト: 8)"
echo "   --max_length: 最大系列長 (デフォルト: 512)"
echo "   --device: cuda, cpu, または自動選択"
echo ""
echo "詳細なヘルプ："
echo "   python main.py --help"
echo ""
echo "注意事項："
echo "- 初回実行時はモデルのダウンロードが行われます"
echo "- GPUを使用する場合は、CUDA環境が正しく設定されていることを確認してください"
echo "- メモリ不足の場合は、batch_sizeやmax_lengthを小さくしてください"
echo "" 