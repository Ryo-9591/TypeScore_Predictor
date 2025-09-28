import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
from datetime import datetime

from app.utils import safe_text_log
from app.logging_config import get_logger, get_report_logger

logger = get_logger(__name__)

# 予測精度レポート用の専用ロガー
report_logger = get_report_logger()


class ModelTrainer:
    """モデル学習のメインクラス"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        モデルトレーナーの初期化

        Args:
            config: モデル設定の辞書
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.metrics = {}
        self.feature_names = []
        self.encoders = {}
        self.training_samples = 0

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルトのモデル設定を取得"""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_splits": 5,
            "target_mae": 200.0,
        }

    def train_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """
        モデルを学習する

        Args:
            X: 特徴量データ
            y: ターゲットデータ

        Returns:
            学習済みモデルと評価指標
        """
        logger.info("モデル学習開始...")

        try:
            # 学習サンプル数を保存
            self.training_samples = len(X)
            logger.debug(f"学習サンプル数: {self.training_samples}")

            # 特徴量名を保存
            self.feature_names = list(X.columns)

            # カテゴリカル特徴量のエンコード
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders = encoders

            # 時系列クロスバリデーション
            train_idx, test_idx = self._perform_time_series_split(X_encoded, y)

            # データの分割
            X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # XGBoostモデルの学習
            model, metrics = self._train_xgboost_model(X_train, y_train, X_test, y_test)

            # モデルとメトリクスを保存
            self.model = model
            self.metrics = metrics

            logger.info(f"モデル学習完了 - テストMAE: {metrics['test_mae']:.2f}")

            # 精度分析レポートの出力
            self._log_accuracy_analysis(X_train, y_train, X_test, y_test, metrics)

            return model, metrics

        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
            self._log_training_error(str(e))
            raise

    def _encode_categorical_features(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """カテゴリカル特徴量をエンコード"""
        logger.info("カテゴリカル特徴量をエンコード中...")

        X_encoded = X.copy()
        encoders = {}

        # カテゴリカル列をエンコード
        categorical_columns = X_encoded.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            logger.info(f"{col}をエンコード: {len(le.classes_)}個のクラス")

        return X_encoded, encoders

    def _perform_time_series_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """時系列クロスバリデーションを実行"""
        logger.info("時系列クロスバリデーションを実行中...")

        tscv = TimeSeriesSplit(n_splits=self.config["n_splits"])
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]

        # データリーク防止（小規模データ用に調整）
        train_size = int(len(train_idx) * 0.9)
        train_idx = train_idx[:train_size]

        logger.info(
            f"学習データ: {len(train_idx)}サンプル, テストデータ: {len(test_idx)}サンプル"
        )
        return train_idx, test_idx

    def _train_xgboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """XGBoostモデルを学習し、評価"""
        logger.info("XGBoostモデルを学習中...")

        # モデル設定
        model_config = {
            "n_estimators": self.config["n_estimators"],
            "max_depth": self.config["max_depth"],
            "learning_rate": self.config["learning_rate"],
            "subsample": self.config["subsample"],
            "colsample_bytree": self.config["colsample_bytree"],
            "random_state": self.config["random_state"],
            "tree_method": "hist",
            "gpu_id": -1,
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**model_config)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False,
        )

        # 予測と評価
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
        }

        logger.info(f"モデル学習完了 - テストMAE: {metrics['test_mae']:.2f}")
        return model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        新しいデータに対して予測を実行

        Args:
            X: 予測用の特徴量データ

        Returns:
            予測結果
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        # カテゴリカル特徴量のエンコード
        X_encoded = self._apply_encoding(X)

        return self.model.predict(X_encoded)

    def _apply_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習時に使用したエンコーディングを適用"""
        X_encoded = X.copy()

        for col, encoder in self.encoders.items():
            if col in X_encoded.columns:
                # 新しいカテゴリの処理
                X_encoded[col] = X_encoded[col].astype(str)
                unknown_mask = ~X_encoded[col].isin(encoder.classes_)
                if unknown_mask.any():
                    # 未知のカテゴリは最頻値で置換
                    most_frequent = encoder.classes_[0]
                    X_encoded.loc[unknown_mask, col] = most_frequent

                X_encoded[col] = encoder.transform(X_encoded[col])

        return X_encoded

    def evaluate_performance(self) -> Dict[str, Any]:
        """
        モデルの性能を評価

        Returns:
            評価結果の辞書
        """
        if not self.metrics:
            raise ValueError("モデルが学習されていません")

        evaluation = {
            "metrics": self.metrics,
            "target_achieved": self.metrics["test_mae"] <= self.config["target_mae"],
            "target_mae": self.config["target_mae"],
            "performance_status": "達成"
            if self.metrics["test_mae"] <= self.config["target_mae"]
            else "未達成",
        }

        logger.info(
            f"性能評価: MAE={self.metrics['test_mae']:.2f}, 目標達成={evaluation['target_achieved']}"
        )
        return evaluation

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        return dict(zip(self.feature_names, self.model.feature_importances_))

    def _log_accuracy_analysis(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Dict[str, float],
    ):
        """精度分析レポートをログに出力"""
        try:
            # 基本統計情報
            train_stats = {
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "min": float(y_train.min()),
                "max": float(y_train.max()),
                "median": float(y_train.median()),
            }

            test_stats = {
                "mean": float(y_test.mean()),
                "std": float(y_test.std()),
                "min": float(y_test.min()),
                "max": float(y_test.max()),
                "median": float(y_test.median()),
            }

            # 特徴量統計
            feature_stats = {}
            for feature in self.feature_names:
                if feature in X_train.columns:
                    feature_stats[feature] = {
                        "train_mean": float(X_train[feature].mean()),
                        "train_std": float(X_train[feature].std()),
                        "test_mean": float(X_test[feature].mean()),
                        "test_std": float(X_test[feature].std()),
                    }

            # 精度分析レポート
            accuracy_report = {
                "event_type": "accuracy_analysis",
                "timestamp": datetime.now().isoformat(),
                "model_config": self.config,
                "data_summary": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                },
                "target_statistics": {"train": train_stats, "test": test_stats},
                "feature_statistics": feature_stats,
                "performance_metrics": metrics,
                "accuracy_assessment": {
                    "mae_target_achieved": bool(
                        metrics["test_mae"] <= self.config["target_mae"]
                    ),
                    "target_mae": self.config["target_mae"],
                    "performance_level": self._assess_performance_level(
                        metrics["test_mae"]
                    ),
                    "improvement_potential": self._assess_improvement_potential(
                        metrics
                    ),
                },
            }

            report_logger.info(
                safe_text_log(accuracy_report, "ACCURACY_ANALYSIS_REPORT")
            )

        except Exception as e:
            logger.error(f"精度分析レポート出力エラー: {e}")

    def _log_training_error(self, error_message: str):
        """学習エラーレポートをログに出力"""
        error_report = {
            "event_type": "training_error",
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "model_config": self.config,
            "training_samples": self.training_samples,
            "feature_count": len(self.feature_names),
        }

        report_logger.error(safe_text_log(error_report, "TRAINING_ERROR_REPORT"))

    def _assess_performance_level(self, mae: float) -> str:
        """パフォーマンスレベルを評価"""
        if mae <= 100:
            return "優秀"
        elif mae <= 200:
            return "良好"
        elif mae <= 400:
            return "普通"
        else:
            return "改善必要"

    def _assess_improvement_potential(
        self, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """改善可能性を評価"""
        mae = metrics["test_mae"]
        rmse = metrics["test_rmse"]

        improvement_potential = {
            "high": mae > 300 or rmse > 500,
            "medium": 200 < mae <= 300 or 300 < rmse <= 500,
            "low": mae <= 200 and rmse <= 300,
            "recommendations": [],
        }

        # 推奨事項の生成
        if mae > 300:
            improvement_potential["recommendations"].append(
                "特徴量エンジニアリングの見直し"
            )
        if rmse > 500:
            improvement_potential["recommendations"].append("モデルパラメータの調整")
        if len(self.feature_names) < 5:
            improvement_potential["recommendations"].append("特徴量の追加")

        return improvement_potential
