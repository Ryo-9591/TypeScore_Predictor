import pandas as pd
import numpy as np


class DataPreprocessor:
    """データの前処理を担当するクラス"""

    def __init__(self, m_user, t_miss, t_score):
        """
        Args:
            m_user (pd.DataFrame): ユーザーデータ
            t_miss (pd.DataFrame): ミスタイプデータ
            t_score (pd.DataFrame): スコアデータ
        """
        self.m_user = m_user
        self.t_miss = t_miss
        self.t_score = t_score
        self.user_mapping = {}
        self.miss_summary = None
        self.merged_data = None

    def preprocess_data(self):
        """データの前処理を実行"""
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

        print("データの前処理が完了しました")
        print(f"結合後データ: {len(self.merged_data)}件")

        return self.merged_data

    def _create_user_mapping(self):
        """ユーザーIDとユーザー名のマッピングを作成"""
        self.user_mapping = dict(zip(self.m_user["user_id"], self.m_user["username"]))

    def _add_usernames_to_scores(self):
        """スコアデータにユーザー名を追加"""
        self.t_score["username"] = self.t_score["user_id"].map(self.user_mapping)

    def _aggregate_miss_data(self):
        """ミスタイプデータをユーザー別に集計"""
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

    def _merge_datasets(self):
        """スコアデータとミスタイプデータを結合"""
        self.merged_data = self.t_score.merge(
            self.miss_summary, on="user_id", how="left"
        ).fillna(0)

    def _create_features(self):
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

    def _create_labels(self):
        """ラベルの作成"""
        # 難易度のラベル
        self.merged_data["difficulty_label"] = self.merged_data["diff_id"].map(
            {1: "Easy", 2: "Normal", 3: "Hard"}
        )

        # 言語のラベル
        self.merged_data["language_label"] = self.merged_data["lang_id"].map(
            {1: "Japanese", 2: "English"}
        )

    def get_merged_data(self):
        """前処理済みの結合データを返す"""
        return self.merged_data

    def get_user_mapping(self):
        """ユーザーマッピングを返す"""
        return self.user_mapping

    def get_feature_columns(self):
        """特徴量の列名を返す"""
        return [
            "diff_id",
            "lang_id",
            "accuracy",
            "typing_count",
            "total_misses",
            "avg_misses",
            "miss_rate",
        ]
