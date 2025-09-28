import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

from app.utils.common import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """特徴量エンジニアリングのメインクラス"""

    def __init__(self):
        """特徴量エンジニアの初期化"""
        self.feature_names = []
        self.feature_stats = {}

    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量を作成して機械学習用のデータセットを生成"""
        logger.info(f"特徴量エンジニアリング開始: {df.shape}")

        try:
            # 特徴量作成のパイプライン
            df_features = df.copy()

            # 特徴量作成ステップ
            feature_steps = [
                ("基本特徴量", self._create_basic_features),
                ("時系列特徴量", self._create_temporal_features),
                ("統計特徴量", self._create_statistical_features),
            ]

            for step_name, step_func in feature_steps:
                logger.info(f"{step_name}作成中...")
                df_features = step_func(df_features)

            # ターゲット変数の分離
            X, y = self._separate_target(df_features)
            self.feature_names = list(X.columns)

            logger.info(
                f"特徴量エンジニアリング完了: {len(self.feature_names)}個の特徴量"
            )
            return X, y

        except Exception as e:
            logger.error(f"特徴量エンジニアリングエラー: {e}")
            raise

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量の作成"""
        df_features = df.copy()

        # ユーザーIDの数値化（ハッシュ化）
        df_features["user_id_numeric"] = (
            df_features["user_id"].astype(str).apply(lambda x: hash(x) % 10000)
        )

        # 比率特徴量の計算（ゼロ除算を避ける）
        df_features["miss_rate"] = np.where(
            df_features["typing_count"] > 0,
            df_features["total_miss"] / df_features["typing_count"],
            0,
        )

        df_features["score_efficiency"] = np.where(
            df_features["typing_count"] > 0,
            df_features["score"] / df_features["typing_count"],
            0,
        )

        return df_features

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量の作成"""
        df_features = df.copy()

        # 日時特徴量
        if "created_at" in df_features.columns:
            df_features["hour"] = df_features["created_at"].dt.hour
            df_features["day_of_week"] = df_features["created_at"].dt.dayofweek
            df_features["is_weekend"] = (
                df_features["day_of_week"].isin([5, 6]).astype(int)
            )

        # ユーザー別の過去スコア特徴量
        df_features = self._create_user_history_features(df_features)

        return df_features

    def _create_user_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ユーザー別の履歴特徴量を作成"""
        df_features = df.copy()

        # ソート処理
        sort_columns = ["user_id"] + (
            ["created_at"] if "created_at" in df_features.columns else []
        )
        df_features = df_features.sort_values(sort_columns)

        # グループ化オブジェクトを事前計算
        user_groups = df_features.groupby("user_id")
        score_groups = user_groups["score"]
        miss_groups = user_groups["total_miss"]

        # 履歴特徴量の一括計算
        history_features = {
            "prev_score": score_groups.shift(1).fillna(df_features["score"].mean()),
            "avg_score_3": score_groups.rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True),
            "max_score_3": score_groups.rolling(3, min_periods=1)
            .max()
            .reset_index(0, drop=True),
            "min_score_3": score_groups.rolling(3, min_periods=1)
            .min()
            .reset_index(0, drop=True),
            "avg_miss_3": miss_groups.rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True),
            "score_std_3": score_groups.rolling(3, min_periods=1)
            .std()
            .reset_index(0, drop=True)
            .fillna(0),
        }

        # 特徴量を一括追加
        for feature_name, feature_values in history_features.items():
            df_features[feature_name] = feature_values

        return df_features

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計特徴量の作成"""
        df_features = df.copy()

        # 正規化処理の関数
        def safe_normalize(group):
            """安全な正規化（標準偏差が0の場合は0を返す）"""
            mean_val = group.mean()
            std_val = group.std()
            return (group - mean_val) / std_val if std_val > 0 else group - mean_val

        # ユーザー別正規化の一括処理
        user_groups = df_features.groupby("user_id")
        df_features["score_normalized"] = (
            user_groups["score"].transform(safe_normalize).fillna(0)
        )
        df_features["miss_normalized"] = (
            user_groups["total_miss"].transform(safe_normalize).fillna(0)
        )

        return df_features

    def _separate_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ターゲット変数の分離"""
        # 特徴量として使用する列を定義
        feature_columns = [
            "user_id_numeric",
            "prev_score",
            "avg_score_3",
            "max_score_3",
            "min_score_3",
            "typing_count",
            "avg_miss_3",
            "score_std_3",
            "miss_rate",
            "score_efficiency",
            "hour",
            "day_of_week",
            "is_weekend",
            "score_normalized",
            "miss_normalized",
        ]

        # 存在する列のみを選択
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].copy()
        y = df["score"].copy()

        # 欠損値の最終処理（数値型のみ）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        return X, y

    def get_feature_names(self) -> List[str]:
        """特徴量名のリストを取得"""
        return self.feature_names.copy()

    def get_feature_importance(self, model) -> Dict[str, float]:
        """
        モデルから特徴量重要度を取得

        Args:
            model: 学習済みモデル

        Returns:
            特徴量重要度の辞書
        """
        if hasattr(model, "feature_importances_"):
            return dict(zip(self.feature_names, model.feature_importances_))
        else:
            logger.warning("モデルに特徴量重要度がありません")
            return {}

    def analyze_feature_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量間の相関を分析

        Args:
            X: 特徴量データフレーム

        Returns:
            相関行列
        """
        return X.corr()
