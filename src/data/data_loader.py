"""
JCommonsenseQAデータセットを読み込むためのデータローダー
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import os
import shutil
import random


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

    def clear_cache(self):
        """HuggingFace Datasetsのキャッシュをクリアする"""
        try:
            import datasets

            datasets.disable_caching()  # キャッシュを無効化
            print("Disabled datasets caching for this session")

            # 既存のキャッシュディレクトリもクリア
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            if os.path.exists(cache_dir):
                print(f"Clearing datasets cache at: {cache_dir}")
                shutil.rmtree(cache_dir)
                print("Cache cleared successfully")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")

    def load_data(self, split: str = "validation") -> List[Dict[str, Any]]:
        """
        HuggingFace Datasetsからデータを読み込む

        Args:
            split: データの分割 ("train" または "validation")

        Returns:
            データのリスト
        """
        # Colab環境での互換性設定
        import os

        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

        try:
            # Colabでの互換性のためキャッシュを無効化
            import datasets

            datasets.disable_caching()

            print(
                f"Loading JCommonsenseQA dataset from HuggingFace (split: {split})..."
            )

            # 直接ダウンロードモードで読み込み（キャッシュを使わない）
            dataset_dict = load_dataset(
                "leemeng/jcommonsenseqa-v1.1"
            )

            print(f"Available splits: {list(dataset_dict.keys())}")
            
            # 指定されたsplitのデータを取得
            if split in dataset_dict:
                dataset = dataset_dict[split]
                print(f"✅ Successfully found '{split}' split with {len(dataset)} samples")
            else:
                print(f"Warning: Split '{split}' not found. Available splits: {list(dataset_dict.keys())}")
                # validationが指定されていてtestがある場合はtestを使用
                if split == "validation" and "test" in dataset_dict:
                    print("Using 'test' split instead of 'validation'")
                    dataset = dataset_dict["test"]
                elif split == "train" and "train" in dataset_dict:
                    print("Using 'train' split")
                    dataset = dataset_dict["train"]
                else:
                    # 最初に見つかったsplitを使用
                    available_split = list(dataset_dict.keys())[0]
                    print(f"Using '{available_split}' split instead")
                    dataset = dataset_dict[available_split]

            # データを辞書のリストに変換
            data = []
            for item in dataset:
                data.append(item)

            print(f"Successfully loaded {len(data)} samples from {split} split")
            
            # データの内容を簡単に確認
            if len(data) > 0:
                first_item = data[0]
                print(f"Sample data structure: {list(first_item.keys())}")
                
            return data

        except Exception as e:
            print(f"❌ Primary loading method failed: {e}")
            print("Trying alternative approaches...")

            # 代替方法1: 別のデータセット名で試行
            try:
                print("Trying alternative dataset: shunk031/JGLUE...")
                import datasets
                datasets.disable_caching()

                dataset = load_dataset(
                    "shunk031/JGLUE",
                    name="JCommonsenseQA",
                    split=split,
                    trust_remote_code=True,
                    verification_mode="no_checks",
                )

                data = []
                for item in dataset:
                    data.append(item)

                print(f"✅ Successfully loaded {len(data)} samples using alternative dataset")
                return data

            except Exception as e2:
                print(f"❌ Alternative dataset loading failed: {e2}")
                print("⚠️  All data loading methods failed. Creating dummy data for testing...")
                return self._create_dummy_data(split)

    def _create_dummy_data(self, split: str) -> List[Dict[str, Any]]:
        """テスト用のダミーデータを作成"""
        dummy_data = []
        # より現実的なサンプル数に増加
        if split == "train":
            num_samples = 1000  # 学習用により多くのサンプル
        elif split == "validation":
            num_samples = 200   # 検証用も十分なサンプル数
        else:
            num_samples = 100

        for i in range(num_samples):
            # ランダムなラベルを生成（規則的なパターンを排除）
            random_label = random.randint(0, 4)
            
            dummy_data.append(
                {
                    "q_id": i,
                    "question": f"テスト質問{i}：これは機械学習のテスト用ダミーデータです。",
                    "choice0": f"選択肢A_{i}",
                    "choice1": f"選択肢B_{i}",
                    "choice2": f"選択肢C_{i}",
                    "choice3": f"選択肢D_{i}",
                    "choice4": f"選択肢E_{i}",
                    "label": random_label,  # ランダムなラベル
                }
            )

        print(f"Created {len(dummy_data)} dummy samples for {split} split")
        print(f"⚠️  注意：これはダミーデータです。実際のデータセット読み込みに失敗しています。")
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
    print("JCommonsenseQA Data Loader Test")

    # 必要に応じてキャッシュをクリア（コメントアウトを外してください）
    # loader = JCommonsenseQALoader()
    # loader.clear_cache()

    loader = JCommonsenseQALoader()
    dataloader = loader.create_dataloader(split="validation", batch_size=2)

    for batch in dataloader:
        print(f"Batch size: {len(batch['choices'])}")
        print(f"Labels: {batch['labels']}")
        print(f"Question IDs: {batch['question_ids']}")
        break
