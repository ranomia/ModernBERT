"""
JCommonsenseQAデータセットを読み込むためのデータローダー
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import os


class JCommonsenseQADataset(Dataset):
    """JCommonsenseQAデータセット用のDatasetクラス"""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 質問と選択肢を結合
        question = item["question"]

        # HuggingFace Datasets形式の選択肢を処理
        choice_texts = []
        for i in range(5):  # choice0 から choice4 まで
            choice_key = f"choice{i}"
            if choice_key in item:
                choice_text = f"{question} {item[choice_key]}"
                choice_texts.append(choice_text)
            else:
                choice_texts.append("")

        # 正解ラベル
        label = int(item["label"])

        # トークン化
        encoded_choices = []
        for text in choice_texts:
            if text:  # 空でない場合のみ
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_choices.append(
                    {
                        "input_ids": encoded["input_ids"].squeeze(),
                        "attention_mask": encoded["attention_mask"].squeeze(),
                    }
                )
            else:
                # 空の場合はパディングトークンで埋める
                encoded_choices.append(
                    {
                        "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                        "attention_mask": torch.zeros(
                            self.max_length, dtype=torch.long
                        ),
                    }
                )

        return {
            "choices": encoded_choices,
            "label": torch.tensor(label, dtype=torch.long),
            "question_id": item.get("q_id", idx),
        }


class JCommonsenseQALoader:
    """JCommonsenseQAデータセットを読み込むクラス"""

    def __init__(self, tokenizer_name: str = "cl-tohoku/bert-base-japanese-v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_data(self, split: str = "validation") -> List[Dict[str, Any]]:
        """
        HuggingFace Datasetsからデータを読み込む

        Args:
            split: データの分割 ("train" または "validation")

        Returns:
            データのリスト
        """
        try:
            # HuggingFace DatasetsからJCommonsenseQAを読み込み
            print(
                f"Loading JCommonsenseQA dataset from HuggingFace (split: {split})..."
            )
            dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA", split=split)

            # データを辞書のリストに変換
            data = []
            for item in dataset:
                data.append(item)

            print(f"Loaded {len(data)} samples from {split} split")
            return data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating dummy data for testing...")
            # テスト用のダミーデータを作成
            return self._create_dummy_data(split)

    def _create_dummy_data(self, split: str) -> List[Dict[str, Any]]:
        """テスト用のダミーデータを作成"""
        dummy_data = []
        num_samples = 10 if split == "validation" else 5

        for i in range(num_samples):
            dummy_data.append(
                {
                    "q_id": i,
                    "question": f"テスト質問{i}",
                    "choice0": "選択肢A",
                    "choice1": "選択肢B",
                    "choice2": "選択肢C",
                    "choice3": "選択肢D",
                    "choice4": "選択肢E",
                    "label": i % 5,
                }
            )

        print(f"Created {len(dummy_data)} dummy samples for {split} split")
        return dummy_data

    def create_dataloader(
        self,
        split: str = "validation",
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = False,
    ) -> DataLoader:
        """DataLoaderを作成する"""
        data = self.load_data(split)
        dataset = JCommonsenseQADataset(data, self.tokenizer, max_length)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """バッチ処理用のcollate関数"""
        batch_choices = []
        batch_labels = []
        batch_ids = []

        for item in batch:
            batch_choices.append(item["choices"])
            batch_labels.append(item["label"])
            batch_ids.append(item["question_id"])

        return {
            "choices": batch_choices,
            "labels": torch.stack(batch_labels),
            "question_ids": batch_ids,
        }


if __name__ == "__main__":
    # テスト用コード
    loader = JCommonsenseQALoader()
    dataloader = loader.create_dataloader(split="validation", batch_size=2)

    for batch in dataloader:
        print(f"Batch size: {len(batch['choices'])}")
        print(f"Labels: {batch['labels']}")
        print(f"Question IDs: {batch['question_ids']}")
        break
