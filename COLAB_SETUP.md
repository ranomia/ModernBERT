# Google Colab 実行ガイド

## 事前設定

Google Colabでこのプロジェクトを実行する前に、以下のセットアップを行ってください。

### 1. 必要なライブラリのインストール

```python
# 必要なライブラリをインストール
!pip install torch transformers datasets accelerate
!pip install --upgrade huggingface_hub

# datasetsライブラリを最新版に更新
!pip install --upgrade datasets
```

### 2. HuggingFace Datasetsの設定

```python
import os
import datasets

# キャッシュを無効化（Colab環境での互換性のため）
datasets.disable_caching()

# 環境変数の設定
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
```

### 3. データセットロードの問題解決

JCommonsenseQAデータセットのロードで問題が発生する場合は、以下を試してください：

```python
# キャッシュクリア
import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("キャッシュをクリアしました")

# datasetsの再インストール
!pip uninstall datasets -y
!pip install datasets
```

## よくある問題と解決方法

### エラー: "Loading a dataset cached in a LocalFileSystem is not supported"

**原因**: Google ColabのHuggingFace Datasetsライブラリのキャッシュ機能の互換性問題

**解決方法**:
1. `datasets.disable_caching()`を実行
2. `download_mode="force_redownload"`を使用
3. ストリーミングモードを使用

### メモリ不足エラー

**解決方法**:
```python
# バッチサイズを小さくする
trainer.quick_finetune(
    tokenizer_name="cl-tohoku/bert-base-japanese-v3",
    num_epochs=2,
    batch_size=2,  # 小さく設定
    max_length=64   # 長さも短く
)

# 定期的にメモリをクリア
import torch
torch.cuda.empty_cache()
```

### GPU使用の確認

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

# GPUメモリ使用量確認
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 推奨実行コード

```python
# 1. 環境設定
import torch
import datasets
datasets.disable_caching()

# 2. デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. モデルとトレーナーの初期化
from models.bert_model import BERTModel
from training.trainer import SimpleTrainer

model = BERTModel()
trainer = SimpleTrainer(
    model=model, 
    device=device, 
    learning_rate=2e-5,
    use_mixed_precision=True if device.type == "cuda" else False
)

# 4. ファインチューニング実行
trainer.quick_finetune(
    tokenizer_name="cl-tohoku/bert-base-japanese-v3",
    num_epochs=2,
    batch_size=4,
    max_length=128
)
```

## パフォーマンス最適化のヒント

1. **混合精度学習**: GPU使用時は`use_mixed_precision=True`を設定
2. **バッチサイズ調整**: メモリ使用量に応じて2-8の範囲で調整
3. **シーケンス長**: 必要最小限の長さ（64-256）に設定
4. **定期的なメモリクリア**: 長時間実行時は`torch.cuda.empty_cache()`を実行 