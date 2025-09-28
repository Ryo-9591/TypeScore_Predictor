"""
モデル学習モジュール
XGBoostモデルの学習、評価、予測を行う
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
import logging
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


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
            print(f"DEBUG: training_samples = {self.training_samples}")

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
            return model, metrics

        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
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

    def create_prediction_plot(
        self, y_test: pd.Series, y_pred: np.ndarray
    ) -> go.Figure:
        """
        予測結果の散布図を作成

        Args:
            y_test: 実測値
            y_pred: 予測値

        Returns:
            PlotlyのFigureオブジェクト
        """
        logger.info("予測結果の散布図を作成中...")

        fig = go.Figure()

        # 散布図の追加
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color="#007bff",
                    line=dict(color="#ffffff", width=1),
                ),
                name="予測値 vs 実測値",
            )
        )

        # 理想的な予測線（y=x）を追加
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#dc3545", width=3, dash="dash"),
                name="理想的な予測線",
            )
        )

        # ダークテーマを適用
        title_text = f"予測スコア vs 実測スコア<br>RMSE: {self.metrics['test_rmse']:.2f}, MAE: {self.metrics['test_mae']:.2f}"
        fig.update_layout(
            title=title_text,
            xaxis_title="実測スコア",
            yaxis_title="予測スコア",
            width=800,
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=16),
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info("予測散布図を作成しました")
        return fig
