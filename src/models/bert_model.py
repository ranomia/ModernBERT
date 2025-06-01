"""
東北大BERT-base v3を使用したBERTモデル
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any
import time
import psutil
import os


class BERTModel(nn.Module):
    """東北大BERT-base v3を使用したモデル"""

    def __init__(
        self, model_name: str = "cl-tohoku/bert-base-japanese-v3", num_choices: int = 5
    ):
        super().__init__()
        self.model_name = model_name
        self.num_choices = num_choices

        # BERTモデルとトークナイザーを読み込み
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 分類ヘッド
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        """
        フォワードパス

        Args:
            input_ids: (batch_size, num_choices, seq_len)
            attention_mask: (batch_size, num_choices, seq_len)

        Returns:
            logits: (batch_size, num_choices)
        """
        batch_size, num_choices, seq_len = input_ids.shape

        # (batch_size * num_choices, seq_len) に変形
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)

        # BERTエンコーディング
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # (batch_size * num_choices, hidden_size)

        # ドロップアウト適用
        pooled_output = self.dropout(pooled_output)

        # 分類
        logits = self.classifier(pooled_output)  # (batch_size * num_choices, 1)

        # (batch_size, num_choices) に変形
        logits = logits.view(batch_size, num_choices)

        return logits

    def predict(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            batch: バッチデータ
            device: デバイス

        Returns:
            予測結果
        """
        self.eval()

        # メモリ使用量測定開始
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # 時間測定開始
        start_time = time.time()

        with torch.no_grad():
            # バッチデータを処理
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
            input_ids = torch.stack(input_ids_list).to(device)
            attention_mask = torch.stack(attention_mask_list).to(device)

            # 予測実行
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

        # 時間測定終了
        end_time = time.time()
        inference_time = end_time - start_time

        # メモリ使用量測定終了
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        return {
            "predictions": predictions.cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "inference_time": inference_time,
            "memory_used": memory_used,
            "batch_size": batch_size,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,  # float32想定
            "hidden_size": self.bert.config.hidden_size,
            "num_layers": self.bert.config.num_hidden_layers,
            "num_attention_heads": self.bert.config.num_attention_heads,
            "vocab_size": self.bert.config.vocab_size,
        }


if __name__ == "__main__":
    # テスト用コード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTModel()
    model.to(device)

    print("BERT Model Info:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # サンプルデータでテスト
    batch_size = 2
    num_choices = 5
    seq_len = 128

    input_ids = torch.randint(0, 1000, (batch_size, num_choices, seq_len))
    attention_mask = torch.ones(batch_size, num_choices, seq_len)

    print(f"\nTest with input shape: {input_ids.shape}")
    logits = model(input_ids.to(device), attention_mask.to(device))
    print(f"Output shape: {logits.shape}")
    print(f"Sample logits: {logits[0].detach().cpu().numpy()}")
