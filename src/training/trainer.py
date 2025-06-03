"""
簡単なファインチューニング機能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import JCommonsenseQALoader


class SimpleTrainer:
    """簡単なファインチューニング用トレーナー"""

    def __init__(self, model, device, learning_rate=2e-5, use_mixed_precision=False):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 混合精度学習用のGradScaler
        if use_mixed_precision and device.type == "cuda":
            self.scaler = GradScaler()
            print("✅ 混合精度学習が有効になりました")
        else:
            self.scaler = None
            if use_mixed_precision and device.type != "cuda":
                print("⚠️  混合精度学習はCUDA環境でのみ使用可能です。無効化されました。")

    def train_epoch(self, dataloader, epoch):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Training Epoch {epoch}")
        ):
            self.optimizer.zero_grad()

            # バッチデータの処理
            batch_size = len(batch["choices"])
            input_ids_list = []
            attention_mask_list = []

            for choices in batch["choices"]:
                choice_input_ids = []
                choice_attention_mask = []

                for choice in choices:
                    choice_input_ids.append(choice["input_ids"])
                    choice_attention_mask.append(choice["attention_mask"])

                input_ids_list.append(torch.stack(choice_input_ids))
                attention_mask_list.append(torch.stack(choice_attention_mask))

            # テンソルに変換してデバイスに移動
            input_ids = torch.stack(input_ids_list).to(self.device)
            attention_mask = torch.stack(attention_mask_list).to(self.device)
            labels = batch["labels"].to(self.device)

            # 混合精度学習対応のフォワードパス
            if self.scaler is not None:
                # 混合精度学習使用
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                # バックワード（混合精度）
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 通常の学習
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # バックワード
                loss.backward()
                self.optimizer.step()

            # 統計更新
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # メモリ使用量制限のため、勾配をクリア
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def evaluate_epoch(self, dataloader):
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # バッチデータの処理
                batch_size = len(batch["choices"])
                input_ids_list = []
                attention_mask_list = []

                for choices in batch["choices"]:
                    choice_input_ids = []
                    choice_attention_mask = []

                    for choice in choices:
                        choice_input_ids.append(choice["input_ids"])
                        choice_attention_mask.append(choice["attention_mask"])

                    input_ids_list.append(torch.stack(choice_input_ids))
                    attention_mask_list.append(torch.stack(choice_attention_mask))

                # テンソルに変換してデバイスに移動
                input_ids = torch.stack(input_ids_list).to(self.device)
                attention_mask = torch.stack(attention_mask_list).to(self.device)
                labels = batch["labels"].to(self.device)

                # フォワードパス（勾配計算なし）
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # 統計更新
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def quick_finetune(
        self, tokenizer_name, num_epochs=3, batch_size=4, max_length=128
    ):
        """クイックファインチューニング"""
        print(f"🚀 クイックファインチューニングを開始します...")
        print(f"エポック数: {num_epochs}, バッチサイズ: {batch_size}")
        print(f"📋 データセット分割戦略:")
        print(f"   - 学習用: trainデータセットの80%")
        print(f"   - 検証用: trainデータセットの20%")
        print(f"   - 最終評価用: validationデータセット（未使用でテスト用に保存）")

        # データローダー作成
        try:
            data_loader_instance = JCommonsenseQALoader(tokenizer_name=tokenizer_name)

            # 訓練データ（trainデータの80%を使用）
            print("📚 学習用データセット（train_split）を読み込み中...")
            train_dataloader = data_loader_instance.create_dataloader(
                split="train_split",  # trainデータセットの学習用分割
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
                validation_split=0.2,  # 20%を検証用に分割
            )
            
            # 検証データ（trainデータの20%を使用）
            print("📖 検証用データセット（val_split）を読み込み中...")
            val_dataloader = data_loader_instance.create_dataloader(
                split="val_split",  # trainデータセットの検証用分割
                batch_size=batch_size,
                max_length=max_length,
                shuffle=False,
                validation_split=0.2,  # 20%を検証用に分割
            )
            
            print("✅ validationデータセットは最終評価用に温存されます")
            
        except Exception as e:
            print(f"⚠️  データローダーの作成に失敗しました: {e}")
            print("ダミーデータで続行します...")

            # エラー時はダミーデータを使用
            data_loader_instance = JCommonsenseQALoader(tokenizer_name=tokenizer_name)
            
            # 学習用ダミーデータ
            train_dataloader = data_loader_instance.create_dataloader(
                split="train",  # ダミーの学習データが作成される
                batch_size=batch_size,
                max_length=max_length,
                shuffle=True,
            )
            
            # 検証用ダミーデータ
            val_dataloader = data_loader_instance.create_dataloader(
                split="validation",  # ダミーの検証データが作成される
                batch_size=batch_size,
                max_length=max_length,
                shuffle=False,
            )

        self.model.to(self.device)

        # 各エポックでファインチューニングと検証を実行
        for epoch in range(1, num_epochs + 1):
            # 学習フェーズ
            avg_loss, train_accuracy = self.train_epoch(train_dataloader, epoch)
            
            # 検証フェーズ
            val_loss, val_accuracy = self.evaluate_epoch(val_dataloader)
            
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

        print("✅ ファインチューニングが完了しました！")
        return self.model


if __name__ == "__main__":
    # テスト用
    from models.bert_model import BERTModel

    device = torch.device("cpu")  # CPU環境用
    model = BERTModel()
    trainer = SimpleTrainer(model, device)

    # ファインチューニング実行（テスト）
    trainer.quick_finetune(
        tokenizer_name="cl-tohoku/bert-base-japanese-v3",
        num_epochs=1,
        batch_size=2,
        max_length=128,
    )
