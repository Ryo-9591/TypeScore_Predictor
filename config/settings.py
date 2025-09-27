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

# モデル設定（過学習を防ぐためより保守的に設定）
MODEL_CONFIG = {
    "n_estimators": 100,     # 木の数を減らす
    "max_depth": 3,          # より浅い木にする
    "learning_rate": 0.1,    # 学習率を上げる
    "subsample": 0.7,        # より強いサブサンプリング
    "colsample_bytree": 0.7, # より強い特徴量サブサンプリング
    "reg_alpha": 1.0,        # より強いL1正則化
    "reg_lambda": 1.0,       # より強いL2正則化
    "min_child_weight": 5,   # 子ノードの最小重みを増やす
    "random_state": 42,
    "n_jobs": -1,
}

# 時系列クロスバリデーション設定
CV_CONFIG = {"n_splits": 5}

# 評価指標の目標値
TARGET_ACCURACY = 50.0  # MAEの目標値（点）- より現実的な値に調整

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
