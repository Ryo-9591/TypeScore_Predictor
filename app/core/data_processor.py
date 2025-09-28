"""
データ処理モジュール
生データの読み込み、クリーニング、前処理を行う
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """データ処理のメインクラス"""

    def __init__(self, data_dir: Path = None):
        """
        データプロセッサーの初期化

        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self._cached_data = None

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        生データを読み込む

        Returns:
            データフレームの辞書
        """
        logger.info("生データを読み込み中...")

        try:
            # CSVファイルを読み込み
            user_df = pd.read_csv(self.data_dir / "m_user.csv")
            miss_df = pd.read_csv(self.data_dir / "t_miss.csv")
            score_df = pd.read_csv(self.data_dir / "t_score.csv")

            logger.info(
                f"データ読み込み完了: user={len(user_df)}, miss={len(miss_df)}, score={len(score_df)}"
            )

            return {"users": user_df, "misses": miss_df, "scores": score_df}

        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

    def clean_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        データをクリーニングして統合

        Args:
            raw_data: 生データの辞書

        Returns:
            クリーニング済みの統合データフレーム
        """
        logger.info("データクリーニング中...")

        try:
            # データの統合
            logger.info("データ統合開始...")
            df_final = self._merge_data(raw_data)
            logger.info(f"データ統合完了: {len(df_final)}行, {len(df_final.columns)}列")

            # データクリーニング
            logger.info("数値データクリーニング開始...")
            df_final = self._clean_numeric_data(df_final)

            logger.info("欠損値処理開始...")
            df_final = self._handle_missing_values(df_final)

            logger.info("日時変換開始...")
            df_final = self._convert_datetime(df_final)

            logger.info(f"データクリーニング完了: {len(df_final)}行")
            return df_final

        except Exception as e:
            logger.error(f"データクリーニングエラー: {e}")
            import traceback

            logger.error(f"詳細エラー: {traceback.format_exc()}")
            raise

    def _merge_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """データを統合"""
        try:
            users_df = raw_data["users"]
            miss_df = raw_data["misses"]
            score_df = raw_data["scores"]

            logger.info(
                f"統合前データ形状: users={users_df.shape}, miss={miss_df.shape}, score={score_df.shape}"
            )
            logger.info(f"score_dfカラム: {list(score_df.columns)}")

            # ミスデータからtotal_missを計算
            miss_totals = miss_df.groupby("user_id")["miss_count"].sum().reset_index()
            miss_totals.rename(columns={"miss_count": "total_miss"}, inplace=True)
            logger.info(f"miss_totals形状: {miss_totals.shape}")

            # スコアデータとミス合計データを統合（created_atの重複を避けるため、必要なカラムのみ選択）
            df_merged = score_df.merge(miss_totals, on="user_id", how="left")

            # デバッグ: マージ後のカラムを確認
            logger.info(f"マージ後のカラム: {list(df_merged.columns)}")

            # ユーザー情報を統合
            df_final = df_merged.merge(users_df, on="user_id", how="left")

            # デバッグ: 最終的なカラムを確認
            logger.info(f"最終的なカラム: {list(df_final.columns)}")

            # total_missの欠損値を0で埋める
            df_final["total_miss"] = df_final["total_miss"].fillna(0)

            return df_final

        except Exception as e:
            logger.error(f"データ統合エラー: {e}")
            import traceback

            logger.error(f"詳細エラー: {traceback.format_exc()}")
            raise

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値データのクリーニング"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # 異常値の処理（外れ値の除去）
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 外れ値を境界値に置換
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の処理"""
        # 数値列の欠損値を中央値で埋める
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # カテゴリ列の欠損値を最頻値で埋める
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """日時データの変換"""
        datetime_columns = ["created_at", "updated_at"]

        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def get_processed_data(self) -> pd.DataFrame:
        """
        処理済みデータを取得（キャッシュ機能付き）

        Returns:
            処理済みデータフレーム
        """
        if self._cached_data is None:
            raw_data = self.load_raw_data()
            self._cached_data = self.clean_data(raw_data)

        return self._cached_data.copy()

    def get_data_info(self) -> Dict[str, Any]:
        """
        データの基本情報を取得

        Returns:
            データ情報の辞書
        """
        df = self.get_processed_data()

        return {
            "total_samples": len(df),
            "unique_users": df["user_id"].nunique(),
            "feature_count": len(df.columns),
            "columns": list(df.columns),
            "score_range": {
                "min": float(df["score"].min()),
                "max": float(df["score"].max()),
                "mean": float(df["score"].mean()),
                "median": float(df["score"].median()),
            },
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }
