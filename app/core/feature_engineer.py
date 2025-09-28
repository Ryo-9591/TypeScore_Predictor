"""
特徴量エンジニアリングモジュール
機械学習用の特徴量を作成・変換する
"""

import pandas as pd
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特徴量エンジニアリングのメインクラス"""

    def __init__(self):
        """特徴量エンジニアの初期化"""
        self.feature_names = []
        self.feature_stats = {}

    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特徴量を作成して機械学習用のデータセットを生成

        Args:
            df: 処理済みデータフレーム

        Returns:
            特徴量データフレームとターゲット変数
        """
        logger.info("特徴量エンジニアリング開始...")

        try:
            logger.info(f"入力データ形状: {df.shape}")
            logger.info(f"入力データカラム: {list(df.columns)}")

            # 基本特徴量の作成
            df_features = self._create_basic_features(df)

            # 時系列特徴量の作成
            df_features = self._create_temporal_features(df_features)

            # 統計特徴量の作成
            df_features = self._create_statistical_features(df_features)

            # ターゲット変数の分離
            X, y = self._separate_target(df_features)

            # 特徴量名を保存
            self.feature_names = list(X.columns)

            logger.info(
                f"特徴量エンジニアリング完了: {len(self.feature_names)}個の特徴量"
            )
            return X, y

        except Exception as e:
            logger.error(f"特徴量エンジニアリングエラー: {e}")
            import traceback

            logger.error(f"詳細エラー: {traceback.format_exc()}")
            raise

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量の作成"""
        df_features = df.copy()

        # ユーザーIDの数値化（ハッシュ化）
        df_features["user_id_numeric"] = (
            df_features["user_id"].astype(str).apply(lambda x: hash(x) % 10000)
        )

        # ミス率の計算
        df_features["miss_rate"] = (
            df_features["total_miss"] / df_features["typing_count"]
        )
        df_features["miss_rate"] = df_features["miss_rate"].fillna(0)

        # スコア効率の計算
        df_features["score_efficiency"] = (
            df_features["score"] / df_features["typing_count"]
        )
        df_features["score_efficiency"] = df_features["score_efficiency"].fillna(0)

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

        # ユーザー別にソート（created_atが存在する場合のみ）
        if "created_at" in df_features.columns:
            df_features = df_features.sort_values(["user_id", "created_at"])
        else:
            # created_atが存在しない場合は、user_idのみでソート
            df_features = df_features.sort_values(["user_id"])
            logger.warning(
                "created_atカラムが存在しません。時系列特徴量をスキップします。"
            )

        # 過去のスコア特徴量
        df_features["prev_score"] = df_features.groupby("user_id")["score"].shift(1)
        df_features["prev_score"] = df_features["prev_score"].fillna(
            df_features["score"].mean()
        )

        # 過去3回の平均スコア
        df_features["avg_score_3"] = (
            df_features.groupby("user_id")["score"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # 過去3回の最大・最小スコア
        df_features["max_score_3"] = (
            df_features.groupby("user_id")["score"]
            .rolling(window=3, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )

        df_features["min_score_3"] = (
            df_features.groupby("user_id")["score"]
            .rolling(window=3, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )

        # 過去3回のミス平均
        df_features["avg_miss_3"] = (
            df_features.groupby("user_id")["total_miss"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # 過去3回のスコア標準偏差
        df_features["score_std_3"] = (
            df_features.groupby("user_id")["score"]
            .rolling(window=3, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        df_features["score_std_3"] = df_features["score_std_3"].fillna(0)

        return df_features

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計特徴量の作成"""
        df_features = df.copy()

        # スコアの正規化（ユーザー別）
        df_features["score_normalized"] = df_features.groupby("user_id")[
            "score"
        ].transform(lambda x: (x - x.mean()) / x.std())
        df_features["score_normalized"] = df_features["score_normalized"].fillna(0)

        # ミス数の正規化
        df_features["miss_normalized"] = df_features.groupby("user_id")[
            "total_miss"
        ].transform(lambda x: (x - x.mean()) / x.std())
        df_features["miss_normalized"] = df_features["miss_normalized"].fillna(0)

        return df_features

    def _separate_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ターゲット変数の分離"""
        # 特徴量として使用する列を選択
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

        # 欠損値の最終処理
        X = X.fillna(X.mean())

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
