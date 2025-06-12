"""
学習曲線を描画するためのプロッタークラス
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# 日本語フォント対応
try:
    import japanize_matplotlib
    print("✅ japanize_matplotlib を読み込みました。日本語表示が有効です。")
except ImportError:
    print("⚠️  japanize_matplotlib が見つかりません。日本語が文字化けする可能性があります。")
    print("   pip install japanize-matplotlib でインストールしてください。")


class LearningCurvePlotter:
    """学習曲線プロッタークラス"""
    
    def __init__(self):
        # 学習履歴を保存する辞書
        self.training_history = {}
        
        # プロットのスタイル設定
        self.colors = {
            'bert_train': '#1f77b4',       # 青（濃）
            'bert_val': '#1f77b4',         # 青（同色、線種で区別）
            'modernbert_train': '#ff7f0e', # オレンジ（濃）
            'modernbert_val': '#ff7f0e',   # オレンジ（同色、線種で区別）
        }
        
        self.line_styles = {
            'train': '-',   # 実線
            'val': '--',    # 破線
        }
        
        self.markers = {
            'bert': 'o',        # 丸
            'modernbert': 's',  # 四角
        }
    
    def initialize_model_history(self, model_name: str) -> None:
        """モデルの学習履歴を初期化"""
        if model_name not in self.training_history:
            self.training_history[model_name] = {
                'epochs': [],
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
    
    def add_epoch_data(self, model_name: str, epoch: int, 
                      train_loss: float, train_accuracy: float,
                      val_loss: float, val_accuracy: float) -> None:
        """エポックデータを追加"""
        self.initialize_model_history(model_name)
        
        history = self.training_history[model_name]
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
    
    def plot_learning_curves(self, save_dir: str = "results", 
                           show_plot: bool = True) -> None:
        """学習曲線をプロット"""
        if not self.training_history:
            print("⚠️  プロットするデータがありません。")
            return
        
        # 保存ディレクトリを作成
        os.makedirs(save_dir, exist_ok=True)
        
        # 2つのサブプロット（正解率とloss）を作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 正解率のプロット
        self._plot_accuracy(ax1)
        
        # lossのプロット
        self._plot_loss(ax2)
        
        # 全体のタイトル
        fig.suptitle('学習曲線比較（BERT vs ModernBERT）', fontsize=16, fontweight='bold')
        
        # 比較モデル数を確認
        model_count = len(self.training_history)
        if model_count > 1:
            print(f"📊 {model_count}つのモデルの学習曲線を同一プロット上に表示します")
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"learning_curves_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 学習曲線を保存しました: {save_path}")
        
        # 表示
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _plot_accuracy(self, ax) -> None:
        """正解率をプロット"""
        ax.set_title('エポックごとの正解率', fontsize=14, fontweight='bold')
        ax.set_xlabel('エポック数')
        ax.set_ylabel('正解率 (%)')
        ax.grid(True, alpha=0.3)
        
        for model_name, history in self.training_history.items():
            if not history['epochs']:
                continue
                
            # モデル名の正規化
            model_type = self._get_model_type(model_name)
            
            # 学習データ（実線）
            train_color = self.colors.get(f'{model_type}_train', '#1f77b4')
            ax.plot(history['epochs'], history['train_accuracy'], 
                   color=train_color, linestyle=self.line_styles['train'],
                   marker=self.markers.get(model_type, 'o'), markersize=6,
                   label=f'{model_name} (学習)', linewidth=2)
            
            # 検証データ（破線）
            val_color = self.colors.get(f'{model_type}_val', '#aec7e8')
            ax.plot(history['epochs'], history['val_accuracy'], 
                   color=val_color, linestyle=self.line_styles['val'],
                   marker=self.markers.get(model_type, 'o'), markersize=6,
                   label=f'{model_name} (推論)', linewidth=2)
        
        ax.legend(loc='best')
        ax.set_ylim([0, 100])
    
    def _plot_loss(self, ax) -> None:
        """lossをプロット"""
        ax.set_title('エポックごとの損失値', fontsize=14, fontweight='bold')
        ax.set_xlabel('エポック数')
        ax.set_ylabel('損失値 (Loss)')
        ax.grid(True, alpha=0.3)
        
        for model_name, history in self.training_history.items():
            if not history['epochs']:
                continue
                
            # モデル名の正規化
            model_type = self._get_model_type(model_name)
            
            # 学習データ（実線）
            train_color = self.colors.get(f'{model_type}_train', '#1f77b4')
            ax.plot(history['epochs'], history['train_loss'], 
                   color=train_color, linestyle=self.line_styles['train'],
                   marker=self.markers.get(model_type, 'o'), markersize=6,
                   label=f'{model_name} (学習)', linewidth=2)
            
            # 検証データ（破線）
            val_color = self.colors.get(f'{model_type}_val', '#aec7e8')
            ax.plot(history['epochs'], history['val_loss'], 
                   color=val_color, linestyle=self.line_styles['val'],
                   marker=self.markers.get(model_type, 'o'), markersize=6,
                   label=f'{model_name} (推論)', linewidth=2)
        
        ax.legend(loc='best')
        ax.set_yscale('log')  # ログスケールでlossを表示
    
    def _get_model_type(self, model_name: str) -> str:
        """モデル名からモデルタイプを取得"""
        model_name_lower = model_name.lower()
        if 'modernbert' in model_name_lower:
            return 'modernbert'
        elif 'bert' in model_name_lower:
            return 'bert'
        else:
            return 'other'
    
    def save_training_history(self, save_dir: str = "results") -> None:
        """学習履歴をJSONファイルに保存"""
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"training_history_{timestamp}.json")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        
        print(f"📝 学習履歴を保存しました: {save_path}")
    
    def print_summary(self) -> None:
        """学習結果のサマリーを出力"""
        if not self.training_history:
            print("⚠️  表示するデータがありません。")
            return
        
        print("\n" + "="*60)
        print("📊 学習結果サマリー")
        print("="*60)
        
        for model_name, history in self.training_history.items():
            if not history['epochs']:
                continue
                
            print(f"\n🤖 {model_name}")
            print(f"   エポック数: {len(history['epochs'])}")
            
            if history['train_accuracy']:
                best_train_acc = max(history['train_accuracy'])
                best_train_epoch = history['epochs'][history['train_accuracy'].index(best_train_acc)]
                print(f"   最高学習正解率: {best_train_acc:.2f}% (エポック {best_train_epoch})")
            
            if history['val_accuracy']:
                best_val_acc = max(history['val_accuracy'])
                best_val_epoch = history['epochs'][history['val_accuracy'].index(best_val_acc)]
                print(f"   最高推論正解率: {best_val_acc:.2f}% (エポック {best_val_epoch})")
            
            if history['train_loss']:
                final_train_loss = history['train_loss'][-1]
                print(f"   最終学習損失: {final_train_loss:.4f}")
            
            if history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                print(f"   最終推論損失: {final_val_loss:.4f}")
        
        print("="*60)
    
    def compare_models(self) -> None:
        """モデル間の比較結果を表示"""
        if len(self.training_history) < 2:
            print("⚠️  比較するには少なくとも2つのモデルが必要です。")
            return
        
        print("\n" + "="*60)
        print("🏆 モデル比較結果")
        print("="*60)
        
        # 最高推論正解率で比較
        best_models = {}
        for model_name, history in self.training_history.items():
            if history['val_accuracy']:
                best_acc = max(history['val_accuracy'])
                best_models[model_name] = best_acc
        
        if best_models:
            sorted_models = sorted(best_models.items(), key=lambda x: x[1], reverse=True)
            print("\n📈 推論正解率ランキング:")
            for i, (model_name, acc) in enumerate(sorted_models, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"   {emoji} {model_name}: {acc:.2f}%")
        
        print("="*60)


if __name__ == "__main__":
    # テスト用コード
    plotter = LearningCurvePlotter()
    
    # サンプルデータを追加
    for epoch in range(1, 6):
        # BERTのサンプルデータ
        plotter.add_epoch_data(
            "BERT", epoch,
            train_loss=1.5 - epoch*0.1,
            train_accuracy=20 + epoch*10,
            val_loss=1.6 - epoch*0.08,
            val_accuracy=18 + epoch*8
        )
        
        # ModernBERTのサンプルデータ
        plotter.add_epoch_data(
            "ModernBERT", epoch,
            train_loss=1.4 - epoch*0.12,
            train_accuracy=25 + epoch*12,
            val_loss=1.5 - epoch*0.1,
            val_accuracy=22 + epoch*10
        )
    
    # プロット作成
    plotter.plot_learning_curves(show_plot=False)
    plotter.print_summary()
    plotter.compare_models() 