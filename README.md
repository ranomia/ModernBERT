# ModernBERT vs BERT 性能検証プロジェクト

このプロジェクトは、ModernBERTとBERTの性能を比較検証するためのコードベースです。

## 概要

- **BERT**: 東北大BERT-base v3 (`cl-tohoku/bert-base-japanese-v3`)
- **ModernBERT**: SB IntuitionsのModernBERT-Ja-130M (`SB-Intuitions/ModernBERT-Ja-130M`)
- **データセット**: JCommonsenseQA
- **タスク**: 日本語常識問題QA

## データセット分割戦略

適切な評価のため、以下のようにデータセットを分割して使用します：

### ファインチューニング時
- **学習用**: JCommonsenseQAの`train`データセットの80%
- **検証用**: JCommonsenseQAの`train`データセットの20%

### 最終評価時
- **テストデータ**: JCommonsenseQAの`validation`データセット
  - 注意：JCommonsenseQAの`test`データセットは非公開のため、`validation`データセットを最終評価用として使用

この分割により、モデルの汎化性能を適切に評価し、データリークを防止します。

## 環境構築

### 1. Python仮想環境の作成

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## プロジェクト構成

```
ModernBERT/
├── README.md
├── requirements.txt
├── .venv/
└── src/
    ├── data/
    │   └── data_loader.py
    ├── models/
    │   ├── bert_model.py
    │   └── modern_bert_model.py
    ├── evaluation/
    │   └── evaluator.py
    ├── utils/
    │   └── metrics.py
    └── main.py
```

## 使用方法

### 1. データの準備とモデルの評価

```bash
cd src
python main.py
```

### 2. 個別の評価

```bash
# BERTの評価
python main.py --model bert

# ModernBERTの評価
python main.py --model modern_bert
```

## 実行結果

実行後、以下の指標で性能が比較されます：

- **Accuracy**: 正解率
- **F1 Score**: F1スコア
- **Inference Time**: 推論時間
- **Memory Usage**: メモリ使用量

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 注意事項

- JCommonsenseQAデータセットの使用には適切なライセンスの確認が必要です
- GPUの使用を推奨します（CPU でも動作しますが処理時間が長くなります）
- 初回実行時はモデルのダウンロードが行われるため、インターネット接続が必要です 