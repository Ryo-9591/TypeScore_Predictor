"""
設定管理モジュール
アプリケーション全体の設定を一元管理
"""

from pathlib import Path
import os

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"

# データファイルの設定
DATA_FILES = {
    "users": DATA_DIR / "m_user.csv",
    "misses": DATA_DIR / "t_miss.csv",
    "scores": DATA_DIR / "t_score.csv",
}

# 出力ファイルの設定
OUTPUT_FILES = {
    "importance_csv": OUTPUT_DIR / "feature_importance.csv",
    "analysis_summary": OUTPUT_DIR / "feature_analysis_summary.txt",
    "prediction_plot": OUTPUT_DIR / "prediction_scatter_plot.html",
    "importance_chart": OUTPUT_DIR / "feature_importance_chart.html",
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

# データベース設定（将来の拡張用）
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "name": os.getenv("DB_NAME", "typescore_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# キャッシュ設定
CACHE_CONFIG = {
    "enabled": True,
    "ttl": 3600,  # 1時間
    "max_size": 1000,
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

# 全設定をまとめた辞書
ALL_CONFIG = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "output_dir": OUTPUT_DIR,
    "config_dir": CONFIG_DIR,
    "data_files": DATA_FILES,
    "output_files": OUTPUT_FILES,
    "model": MODEL_CONFIG,
    "cv": CV_CONFIG,
    "target_accuracy": TARGET_ACCURACY,
    "log": LOG_CONFIG,
    "dashboard": DASHBOARD_CONFIG,
    "api": API_CONFIG,
    "database": DATABASE_CONFIG,
    "cache": CACHE_CONFIG,
    "security": SECURITY_CONFIG,
    "environment": ENVIRONMENT,
    "debug": DEBUG,
}
