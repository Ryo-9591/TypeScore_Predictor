"""
設定管理モジュール
アプリケーション全体の設定を一元管理
"""

from pathlib import Path
import os

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# データファイルの設定
DATA_FILES = {
    "users": DATA_DIR / "m_user.csv",
    "misses": DATA_DIR / "t_miss.csv",
    "scores": DATA_DIR / "t_score.csv",
}

# モデル設定
MODEL_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# クロスバリデーション設定
CV_CONFIG = {
    "n_splits": 5,
    "test_size": 0.2,
}

# 目標精度設定
TARGET_ACCURACY = 200.0

# ログ設定
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log",
    "rotation": "daily",  # 日次ローテーション
    "retention": 30,  # 30日間保持
}

# 予測精度レポート用ログ設定（日付付きファイルのみ）
PREDICTION_REPORT_CONFIG = {
    "enabled": True,
    "file_pattern": PROJECT_ROOT
    / "logs"
    / "prediction_report_{date}.log",  # 日付付きパターン
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "level": "INFO",
    "rotation": "daily",  # 日次ローテーション
    "retention": 30,  # 30日間保持
    "include_metrics": True,  # メトリクス情報を含める
    "include_feature_importance": True,  # 特徴量重要度を含める
    "include_prediction_details": True,  # 予測詳細を含める
}

# ダッシュボード設定
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 8050,
    "debug": True,
    "title": "TypeScore Predictor Dashboard",
}

# API設定
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "TypeScore Predictor API",
    "description": "タイピングスコア予測のためのREST API",
    "version": "1.0.0",
}


# セキュリティ設定
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"],
}

# 環境設定
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
