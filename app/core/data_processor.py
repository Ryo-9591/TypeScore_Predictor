import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from app.utils.common import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """データ処理のメインクラス"""

    def __init__(self, data_dir: Path = None):
        """データプロセッサーの初期化"""
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self._cached_data = None

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """生データを読み込む"""
        logger.info("生データを読み込み中...")

        try:
            # CSVファイルを読み込み
            files = {
                "users": self.data_dir / "m_user.csv",
                "misses": self.data_dir / "t_miss.csv",
                "scores": self.data_dir / "t_score.csv",
            }

            data = {name: pd.read_csv(path) for name, path in files.items()}

            logger.info(
                f"データ読み込み完了: {', '.join([f'{k}={len(v)}' for k, v in data.items()])}"
            )
            return data

        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

    def clean_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """データをクリーニングして統合"""
        logger.info("データクリーニング中...")

        try:
            # データの統合とクリーニング
            df_final = self._merge_data(raw_data)
            logger.info(f"データ統合完了: {len(df_final)}行, {len(df_final.columns)}列")

            # クリーニング処理を順次実行
            cleaning_steps = [
                ("数値データクリーニング", self._clean_numeric_data),
                ("欠損値処理", self._handle_missing_values),
                ("日時変換", self._convert_datetime),
            ]

            for step_name, step_func in cleaning_steps:
                logger.info(f"{step_name}開始...")
                df_final = step_func(df_final)

            logger.info(f"データクリーニング完了: {len(df_final)}行")
            return df_final

        except Exception as e:
            logger.error(f"データクリーニングエラー: {e}")
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

            # スコアデータとミス合計データを統合
            df_merged = score_df.merge(miss_totals, on="user_id", how="left")

            # デバッグ: マージ後のカラムを確認
            logger.info(f"マージ後のカラム: {list(df_merged.columns)}")

            # ユーザー情報を統合（created_atの重複を避けるため、suffixesを使用）
            df_final = df_merged.merge(
                users_df, on="user_id", how="left", suffixes=("", "_user")
            )

            # created_at_xとcreated_at_yが存在する場合は、created_at_xを優先してcreated_atに統一
            if (
                "created_at_x" in df_final.columns
                and "created_at_y" in df_final.columns
            ):
                df_final["created_at"] = df_final["created_at_x"]
                df_final = df_final.drop(columns=["created_at_x", "created_at_y"])
            elif "created_at_x" in df_final.columns:
                df_final["created_at"] = df_final["created_at_x"]
                df_final = df_final.drop(columns=["created_at_x"])
            elif "created_at_y" in df_final.columns:
                df_final["created_at"] = df_final["created_at_y"]
                df_final = df_final.drop(columns=["created_at_y"])

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
        """数値データのクリーニング（外れ値処理）"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # IQR法による外れ値のクリッピング
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            bounds = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(*bounds)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の処理"""
        # 数値列の欠損値を中央値で埋める
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # カテゴリ列の欠損値を最頻値で埋める
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(
                    df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "unknown"
                )

        return df

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """日時データの変換"""
        datetime_columns = ["created_at", "updated_at"]

        for col in datetime_columns:
            if col in df.columns:
                # UTC時間を日本時間に変換してからdatetime型に変換
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

                # UTCから日本時間（JST）に変換
                df[col] = df[col].dt.tz_convert("Asia/Tokyo")

        return df

    def processed_data(self) -> pd.DataFrame:
        """処理済みデータを取得（キャッシュ機能付き）"""
        if self._cached_data is None:
            raw_data = self.load_raw_data()
            self._cached_data = self.clean_data(raw_data)
        return self._cached_data.copy()

    def get_processed_data(self) -> pd.DataFrame:
        """処理済みデータを取得（互換性のため）"""
        return self.processed_data()

    def get_data_info(self) -> Dict[str, Any]:
        """データの基本情報を取得"""
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
            "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        }
