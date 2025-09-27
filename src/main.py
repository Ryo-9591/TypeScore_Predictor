"""
TypeScore Predictor - メイン実行スクリプト
タイピングスコア予測システムの統合実行

実行手順:
1. データ準備とクリーニング
2. 特徴量エンジニアリング（時系列処理）
3. モデル学習と評価
4. 特徴量重要度分析
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 自作モジュールのインポート
from .data_preparation import prepare_data
from .feature_engineering import engineer_features
from .model_training import train_and_evaluate_model
from .feature_importance_analysis import analyze_feature_importance
from config import OUTPUT_DIR, OUTPUT_FILES, TARGET_ACCURACY


def create_output_directory():
    """出力ディレクトリを作成する"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"出力ディレクトリ '{OUTPUT_DIR}' を作成しました")


def print_execution_summary(
    start_time: datetime,
    end_time: datetime,
    metrics: dict,
    feature_count: int,
    sample_count: int,
):
    """実行サマリーを出力する"""
    execution_time = end_time - start_time

    print("\n" + "=" * 60)
    print("           TypeScore Predictor 実行完了")
    print("=" * 60)
    print(f"実行時間: {execution_time.total_seconds():.2f}秒")
    print(f"処理サンプル数: {sample_count:,}件")
    print(f"特徴量数: {feature_count}個")
    print(f"テストデータ RMSE: {metrics['test_rmse']:.2f}")
    print(f"テストデータ MAE: {metrics['test_mae']:.2f}")

    # 精度目標との比較
    if metrics["test_mae"] <= TARGET_ACCURACY:
        print(f"✅ 目標精度 ({TARGET_ACCURACY}点以内) を達成しました")
    else:
        print(f"❌ 目標精度 ({TARGET_ACCURACY}点以内) を未達成です")

    print("\n生成されたファイル:")
    output_files = [
        OUTPUT_DIR / OUTPUT_FILES["scatter_plot"],
        OUTPUT_DIR / OUTPUT_FILES["importance_chart"],
        OUTPUT_DIR / OUTPUT_FILES["importance_csv"],
        OUTPUT_DIR / OUTPUT_FILES["analysis_summary"],
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (生成されませんでした)")

    print("=" * 60)


def main():
    """メイン実行関数"""
    start_time = datetime.now()

    print("TypeScore Predictor を開始します...")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 出力ディレクトリの作成
        create_output_directory()

        # 1. データ準備とクリーニング
        print("\n【ステップ1】データ準備とクリーニング")
        print("-" * 50)
        df_final = prepare_data()

        # データの基本情報を表示
        print(f"\nデータの基本情報:")
        print(f"  行数: {df_final.shape[0]:,}")
        print(f"  列数: {df_final.shape[1]}")
        print(f"  ユニークユーザー数: {df_final['user_id'].nunique()}")

        # 2. 特徴量エンジニアリング
        print("\n【ステップ2】特徴量エンジニアリング")
        print("-" * 50)
        X, y = engineer_features(df_final)

        print(f"\n特徴量エンジニアリング結果:")
        print(f"  特徴量数: {X.shape[1]}")
        print(f"  サンプル数: {X.shape[0]:,}")
        print(f"  特徴量名: {list(X.columns)}")

        # 3. モデル学習と評価
        print("\n【ステップ3】モデル学習と評価")
        print("-" * 50)
        model, metrics = train_and_evaluate_model(X, y)

        # 4. 特徴量重要度分析
        print("\n【ステップ4】特徴量重要度分析")
        print("-" * 50)
        analyze_feature_importance(model, list(X.columns))

        # 実行サマリーの出力
        end_time = datetime.now()
        print_execution_summary(start_time, end_time, metrics, X.shape[1], X.shape[0])

        print("\n🎉 TypeScore Predictor の実行が正常に完了しました！")

    except FileNotFoundError as e:
        print(f"\n❌ エラー: ファイルが見つかりません - {e}")
        print("data/ ディレクトリに以下のファイルが存在することを確認してください:")
        print("  - t_score.csv")
        print("  - t_miss.csv")
        print("  - m_user.csv")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        print("エラーの詳細:")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
