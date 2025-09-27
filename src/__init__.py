"""
TypeScore Predictor - タイピングスコア予測システム

このパッケージは、過去のタイピングセッションデータから次のセッションのスコアを予測する
XGBoostベースの機械学習システムを提供します。
"""

__version__ = "1.0.0"
__author__ = "TypeScore Predictor Team"
__email__ = "contact@typescore-predictor.com"

from .data_preparation import prepare_data
from .feature_engineering import engineer_features
from .model_training import train_and_evaluate_model
from .feature_importance_analysis import analyze_feature_importance

__all__ = [
    "prepare_data",
    "engineer_features",
    "train_and_evaluate_model",
    "analyze_feature_importance",
]
