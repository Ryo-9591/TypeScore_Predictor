import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from logger import logger
from config import config


class DataPreprocessor:
    """データの前処理を担当するクラス"""

    def __init__(
        self, m_user: pd.DataFrame, t_miss: pd.DataFrame, t_score: pd.DataFrame
    ):
        """
        Args:
            m_user (pd.DataFrame): ユーザーデータ
            t_miss (pd.DataFrame): ミスタイプデータ
            t_score (pd.DataFrame): スコアデータ
        """
        self.m_user = m_user
        self.t_miss = t_miss
        self.t_score = t_score
        self.user_mapping: Dict[str, str] = {}
        self.miss_summary: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None

    def preprocess_data(self) -> pd.DataFrame:
        """データの前処理を実行

        Returns:
            pd.DataFrame: 前処理済みの結合データ
        """
        try:
            logger.info("データの前処理を開始します...")

            # ユーザー名のマッピング
            self._create_user_mapping()

            # スコアデータにユーザー名を追加
            self._add_usernames_to_scores()

            # ミスタイプデータをユーザー別に集計
            self._aggregate_miss_data()

            # データの結合
            self._merge_datasets()

            # 特徴量の作成
            self._create_features()

            # ラベルの作成
            self._create_labels()

            # データの検証
            if not self._validate_processed_data():
                raise ValueError("前処理後のデータ検証に失敗しました")

            logger.info(f"データの前処理が完了しました: {len(self.merged_data)}件")
            return self.merged_data

        except Exception as e:
            logger.exception(f"データの前処理中にエラーが発生しました: {e}")
            raise

    def _create_user_mapping(self) -> None:
        """ユーザーIDとユーザー名のマッピングを作成"""
        self.user_mapping = dict(zip(self.m_user["user_id"], self.m_user["username"]))
        logger.debug(f"ユーザーマッピング作成完了: {len(self.user_mapping)}件")

    def _add_usernames_to_scores(self) -> None:
        """スコアデータにユーザー名を追加"""
        self.t_score["username"] = self.t_score["user_id"].map(self.user_mapping)
        logger.debug("スコアデータにユーザー名を追加しました")

    def _aggregate_miss_data(self) -> None:
        """ミスタイプデータをユーザー別に集計"""
        if len(self.t_miss) == 0:
            logger.warning("ミスタイプデータが空です")
            self.miss_summary = pd.DataFrame(
                columns=[
                    "user_id",
                    "total_misses",
                    "avg_misses",
                    "std_misses",
                    "miss_types",
                ]
            )
            return

        self.miss_summary = (
            self.t_miss.groupby("user_id")
            .agg({"miss_count": ["sum", "mean", "std", "count"]})
            .round(2)
        )
        self.miss_summary.columns = [
            "total_misses",
            "avg_misses",
            "std_misses",
            "miss_types",
        ]
        self.miss_summary = self.miss_summary.reset_index()
        logger.debug(f"ミスタイプデータ集計完了: {len(self.miss_summary)}件")

    def _merge_datasets(self) -> None:
        """スコアデータとミスタイプデータを結合"""
        self.merged_data = self.t_score.merge(
            self.miss_summary, on="user_id", how="left"
        ).fillna(0)
        logger.debug(f"データセット結合完了: {len(self.merged_data)}件")

    def _create_features(self) -> None:
        """特徴量の作成"""
        # ミスタイプ率の計算（0除算を防ぐ）
        self.merged_data["miss_rate"] = np.where(
            self.merged_data["typing_count"] > 0,
            self.merged_data["total_misses"] / self.merged_data["typing_count"],
            0,
        )
        self.merged_data["miss_rate"] = self.merged_data["miss_rate"].fillna(0)

        # 無限大やNaNを0で置換
        self.merged_data = self.merged_data.replace([np.inf, -np.inf], 0)
        self.merged_data = self.merged_data.fillna(0)

        # その他の特徴量（必要に応じて追加）
        self.merged_data["typing_speed"] = (
            self.merged_data["typing_count"] / 60  # 仮想的なタイピング速度
        )

        logger.debug("特徴量の作成が完了しました")

    def _create_labels(self) -> None:
        """ラベルの作成"""
        # 難易度のラベル
        self.merged_data["difficulty_label"] = self.merged_data["diff_id"].map(
            {1: "Easy", 2: "Normal", 3: "Hard"}
        )

        # 言語のラベル
        self.merged_data["language_label"] = self.merged_data["lang_id"].map(
            {1: "Japanese", 2: "English"}
        )

        logger.debug("ラベルの作成が完了しました")

    def _validate_processed_data(self) -> bool:
        """前処理後のデータを検証

        Returns:
            bool: 検証成功の場合True
        """
        if self.merged_data is None or len(self.merged_data) == 0:
            logger.error("前処理後のデータが空です")
            return False

        # 必要なカラムの存在確認
        required_columns = set(config.features.feature_columns) | {
            "username",
            "difficulty_label",
            "language_label",
            "score",
        }
        missing_columns = required_columns - set(self.merged_data.columns)
        if missing_columns:
            logger.error(f"必要なカラムが不足しています: {missing_columns}")
            return False

        # NaN値のチェック
        nan_counts = self.merged_data.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN値が存在します: {nan_counts[nan_counts > 0].to_dict()}")

        logger.debug("前処理後のデータ検証が完了しました")
        return True

    def get_merged_data(self) -> Optional[pd.DataFrame]:
        """前処理済みの結合データを返す

        Returns:
            Optional[pd.DataFrame]: 前処理済みデータ
        """
        return self.merged_data

    def get_user_mapping(self) -> Dict[str, str]:
        """ユーザーマッピングを返す

        Returns:
            Dict[str, str]: ユーザーマッピング辞書
        """
        return self.user_mapping

    def get_feature_columns(self) -> List[str]:
        """特徴量の列名を返す

        Returns:
            List[str]: 特徴量の列名リスト
        """
        return config.features.feature_columns.copy()

    def get_data_summary(self) -> Dict[str, any]:
        """データのサマリー情報を返す

        Returns:
            Dict[str, any]: サマリー情報
        """
        if self.merged_data is None:
            return {"error": "データが処理されていません"}

        return {
            "total_records": len(self.merged_data),
            "unique_users": self.merged_data["user_id"].nunique(),
            "unique_modes": len(self.merged_data.groupby(["diff_id", "lang_id"])),
            "score_range": {
                "min": float(self.merged_data["score"].min()),
                "max": float(self.merged_data["score"].max()),
                "mean": float(self.merged_data["score"].mean()),
            },
            "accuracy_range": {
                "min": float(self.merged_data["accuracy"].min()),
                "max": float(self.merged_data["accuracy"].max()),
                "mean": float(self.merged_data["accuracy"].mean()),
            },
        }
