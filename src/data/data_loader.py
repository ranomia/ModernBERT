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
            dataset = load_dataset(
                "leemeng/jcommonsenseqa-v1.1"
            )

            # データを辞書のリストに変換
            data = []
            for item in dataset:
                data.append(item)

            print(f"Successfully loaded {len(data)} samples from {split} split")
            return data

        except Exception as e:
            print(f"Primary loading method failed: {e}")
            print("Trying streaming approach...")

            # 代替方法: ストリーミングモード
            try:
                import datasets

                datasets.disable_caching()

                dataset = load_dataset(
                    "shunk031/JGLUE",
                    name="JCommonsenseQA",
                    split=split,
                    streaming=True,  # ストリーミングモード
                    trust_remote_code=True,  # リモートコードを信頼
                    verification_mode="no_checks",
                )

                # ストリーミングデータを辞書リストに変換
                data = []
                for i, item in enumerate(dataset):
                    if i >= 1000:  # メモリ節約のため制限
                        break
                    data.append(item)

                print(f"Successfully loaded {len(data)} samples using streaming mode")
                return data

            except Exception as e2:
                print(f"Streaming method also failed: {e2}")
                print("Trying alternative dataset approach...")

                # 代替方法3: 別のアプローチでデータセットを取得
                try:
                    # 手動でキャッシュ設定
                    import datasets
                    from datasets import DownloadConfig

                    datasets.disable_caching()

                    download_config = DownloadConfig(
                        force_download=True,
                        resume_download=False,
                        use_etag=False,
                    )

                    # 直接的なアプローチ
                    dataset = load_dataset(
                        "shunk031/JGLUE",
                        "JCommonsenseQA",
                        split=split,
                        trust_remote_code=True,
                        download_config=download_config,
                    )

                    data = []
                    for item in dataset:
                        data.append(item)

                    print(
                        f"Successfully loaded {len(data)} samples using alternative approach"
                    )
                    return data

                except Exception as e3:
                    print(f"Alternative approach also failed: {e3}")
                    print("Trying minimal approach...")

                    # 代替方法4: 最小限のアプローチ
                    try:
                        # キャッシュディレクトリをクリア
                        self.clear_cache()

                        import datasets

                        datasets.disable_caching()

                        # 最小限の設定で試行
                        dataset = load_dataset(
                            "shunk031/JGLUE",
                            name="JCommonsenseQA",
                            trust_remote_code=True,  # リモートコードを信頼
                            verification_mode="no_checks",
                        )[split]

                        data = []
                        for item in dataset:
                            data.append(item)

                        print(
                            f"Successfully loaded {len(data)} samples using minimal approach"
                        )
                        return data

                    except Exception as e4:
                        print(f"All loading methods failed: {e4}")
                        print("Creating dummy data for testing...")
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
