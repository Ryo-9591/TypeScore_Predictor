"""
設定ファイル
アプリケーションの設定値を管理する
"""

import os
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# データファイル名
SCORE_FILE = "t_score.csv"
MISS_FILE = "t_miss.csv"
USER_FILE = "m_user.csv"

# モデル設定（小規模データ用に調整）
MODEL_CONFIG = {
    "n_estimators": 50,  # データ量に合わせて木の数を減らす
    "max_depth": 2,  # より浅い木で過学習を防ぐ
    "learning_rate": 0.05,  # より低い学習率
    "subsample": 0.8,  # サブサンプリング
    "colsample_bytree": 0.8,  # 特徴量サブサンプリング
    "reg_alpha": 0.1,  # L1正則化
    "reg_lambda": 0.1,  # L2正則化
    "min_child_weight": 3,  # 子ノードの最小重み
    "random_state": 42,
    "n_jobs": -1,
}

# 時系列クロスバリデーション設定
CV_CONFIG = {"n_splits": 5}

# 評価指標の目標値
TARGET_ACCURACY = 200.0  # MAEの目標値（点）- 小規模データに現実的な値に調整

# 特徴量エンジニアリング設定
FEATURE_CONFIG = {
    "rolling_window": 3,
    "time_series_features": [
        "prev_score",
        "avg_score_3",
        "avg_miss_3",
        "score_std_3",
        "max_score_3",
        "min_score_3",
    ],
}

# 出力ファイル名
OUTPUT_FILES = {
    "scatter_plot": "prediction_scatter_plot.html",
    "importance_chart": "feature_importance_chart.html",
    "importance_csv": "feature_importance.csv",
    "analysis_summary": "feature_analysis_summary.txt",
}

# ログ設定
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
