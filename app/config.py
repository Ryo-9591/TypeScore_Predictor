"""
アプリケーション設定ファイル
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """モデル訓練設定"""

    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    test_size: float = 0.2
    random_state: int = 42
    min_data_threshold: int = 10


@dataclass
class AppConfig:
    """アプリケーション設定"""

    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    data_path: str = "data/"
    log_level: str = "INFO"
    theme: str = "DARKLY"


@dataclass
class FeatureConfig:
    """特徴量設定"""

    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                "diff_id",
                "lang_id",
                "accuracy",
                "typing_count",
                "total_misses",
                "avg_misses",
                "miss_rate",
            ]


class Config:
    """設定管理クラス"""

    def __init__(self):
        self.model = ModelConfig()
        self.app = AppConfig()
        self.features = FeatureConfig()

        # 環境変数から設定を読み込み
        self._load_from_env()

    def _load_from_env(self):
        """環境変数から設定を読み込み"""
        self.app.host = os.getenv("APP_HOST", self.app.host)
        self.app.port = int(os.getenv("APP_PORT", self.app.port))
        self.app.debug = os.getenv("APP_DEBUG", "false").lower() == "true"
        self.app.data_path = os.getenv("DATA_PATH", self.app.data_path)
        self.app.log_level = os.getenv("LOG_LEVEL", self.app.log_level)

        # モデル設定
        self.model.n_estimators = int(
            os.getenv("MODEL_N_ESTIMATORS", self.model.n_estimators)
        )
        self.model.max_depth = int(os.getenv("MODEL_MAX_DEPTH", self.model.max_depth))
        self.model.min_samples_split = int(
            os.getenv("MODEL_MIN_SAMPLES_SPLIT", self.model.min_samples_split)
        )
        self.model.test_size = float(os.getenv("MODEL_TEST_SIZE", self.model.test_size))
        self.model.random_state = int(
            os.getenv("MODEL_RANDOM_STATE", self.model.random_state)
        )
        self.model.min_data_threshold = int(
            os.getenv("MODEL_MIN_DATA_THRESHOLD", self.model.min_data_threshold)
        )


# グローバル設定インスタンス
config = Config()
