"""
簡単なファインチューニング機能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import JCommonsenseQALoader


class SimpleTrainer:
    """簡単なファインチューニング用トレーナー"""

    def __init__(self, model, device, learning_rate=2e-5):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

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

            # フォワードパス
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

    def quick_finetune(
        self, tokenizer_name, num_epochs=3, batch_size=4, max_length=128
    ):
        """クイックファインチューニング"""
        print(f"🚀 クイックファインチューニングを開始します...")
        print(f"エポック数: {num_epochs}, バッチサイズ: {batch_size}")

        # データローダー作成
        data_loader_instance = JCommonsenseQALoader(tokenizer_name=tokenizer_name)

        # 訓練データ（validationデータの一部を使用）
        train_dataloader = data_loader_instance.create_dataloader(
            split="validation",
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True,
        )

        self.model.to(self.device)

        for epoch in range(1, num_epochs + 1):
            avg_loss, accuracy = self.train_epoch(train_dataloader, epoch)
            print(
                f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
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
