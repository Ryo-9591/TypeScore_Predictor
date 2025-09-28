import pandas as pd
import os
from typing import Tuple, Optional, Dict
from logger import logger
from config import config


class DataLoader:
    """データの読み込みを担当するクラス"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path (str, optional): データファイルのパス
        """
        self.data_path = data_path or config.app.data_path
        self.m_user: Optional[pd.DataFrame] = None
        self.t_miss: Optional[pd.DataFrame] = None
        self.t_score: Optional[pd.DataFrame] = None
        self._user_mapping: Optional[Dict[str, str]] = None

    def load_data(self) -> bool:
        """すべてのデータファイルを読み込む

        Returns:
            bool: 読み込み成功の場合True
        """
        try:
            # データファイルの存在確認
            required_files = ["m_user.csv", "t_miss.csv", "t_score.csv"]
            for file in required_files:
                file_path = os.path.join(self.data_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"データファイルが見つかりません: {file_path}")
                    return False

            # データの読み込み（型指定とメモリ最適化）
            self.m_user = pd.read_csv(
                os.path.join(self.data_path, "m_user.csv"),
                dtype={"user_id": "string", "username": "string"},
            )
            self.t_miss = pd.read_csv(
                os.path.join(self.data_path, "t_miss.csv"),
                dtype={"user_id": "string", "miss_count": "int32"},
            )
            self.t_score = pd.read_csv(
                os.path.join(self.data_path, "t_score.csv"),
                dtype={
                    "user_id": "string",
                    "diff_id": "int8",
                    "lang_id": "int8",
                    "score": "float32",
                    "accuracy": "float32",
                    "typing_count": "int32",
                },
            )

            logger.info("データの読み込みが完了しました")
            logger.info(f"ユーザーデータ: {len(self.m_user)}件")
            logger.info(f"ミスタイプデータ: {len(self.t_miss)}件")
            logger.info(f"スコアデータ: {len(self.t_score)}件")

            return True

        except Exception as e:
            logger.exception(f"データの読み込みに失敗しました: {e}")
            return False

    def get_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """読み込んだデータを返す

        Returns:
            Tuple: (m_user, t_miss, t_score) のタプル
        """
        return self.m_user, self.t_miss, self.t_score

    def get_user_mapping(self) -> Dict[str, str]:
        """ユーザーIDとユーザー名のマッピングを返す

        Returns:
            Dict[str, str]: ユーザーIDをキー、ユーザー名を値とする辞書
        """
        if self.m_user is not None and self._user_mapping is None:
            self._user_mapping = dict(
                zip(self.m_user["user_id"], self.m_user["username"])
            )
        return self._user_mapping or {}

    def validate_data(self) -> bool:
        """データの整合性をチェック

        Returns:
            bool: データが有効な場合True
        """
        if not all(
            [self.m_user is not None, self.t_miss is not None, self.t_score is not None]
        ):
            logger.error("データが読み込まれていません")
            return False

        # 基本的なデータチェック
        if len(self.m_user) == 0:
            logger.error("ユーザーデータが空です")
            return False

        if len(self.t_score) == 0:
            logger.error("スコアデータが空です")
            return False

        # 必要なカラムの存在チェック
        required_columns = {
            "m_user": ["user_id", "username"],
            "t_miss": ["user_id", "miss_count"],
            "t_score": [
                "user_id",
                "diff_id",
                "lang_id",
                "score",
                "accuracy",
                "typing_count",
            ],
        }

        for df_name, columns in required_columns.items():
            df = getattr(self, df_name)
            missing_columns = set(columns) - set(df.columns)
            if missing_columns:
                logger.error(
                    f"{df_name}に必要なカラムが不足しています: {missing_columns}"
                )
                return False

        logger.info("データの整合性チェックが完了しました")
        return True
