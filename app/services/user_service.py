import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core import DataProcessor
from app.utils.common import get_logger, get_jst_time

logger = get_logger(__name__)


class UserService:
    """ユーザーサービスのメインクラス"""

    def __init__(self):
        """ユーザーサービスの初期化"""
        self.data_processor = DataProcessor()

    def get_all_users(self) -> List[str]:
        """全ユーザーのリストを取得"""
        try:
            df = self._get_cached_data()
            users = sorted([str(user_id) for user_id in df["user_id"].unique()])

            logger.info(f"ユーザー一覧取得: {len(users)}人")
            return users

        except Exception as e:
            logger.error(f"ユーザー一覧取得エラー: {e}")
            return []

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """指定されたユーザーの統計データを取得"""
        try:
            df = self._get_cached_data()
            user_data = df[df["user_id"].astype(str) == user_id]

            if len(user_data) == 0:
                logger.warning(f"ユーザーが見つかりません: {user_id}")
                return None

            # 統計計算
            stats = {
                "user_id": user_id,
                "total_sessions": len(user_data),
                "avg_score": float(user_data["score"].mean()),
                "max_score": float(user_data["score"].max()),
                "min_score": float(user_data["score"].min()),
                "latest_score": float(user_data["score"].iloc[-1]),
                "trend": "stable",
            }

            logger.info(
                f"ユーザー統計取得完了: {user_id}, セッション数={stats['total_sessions']}"
            )
            return stats

        except Exception as e:
            logger.error(f"ユーザー統計取得エラー: {user_id}, {e}")
            return None

    def get_user_timeseries(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        指定されたユーザーの時系列データを取得

        Args:
            user_id: ユーザーID

        Returns:
            時系列データの辞書
        """
        try:
            df = self._get_cached_data()
            user_data = df[df["user_id"].astype(str) == user_id]

            if len(user_data) == 0:
                logger.warning(f"ユーザーが見つかりません: {user_id}")
                return None

            # 時系列データをソート（created_atが確実に存在することを前提）
            # created_at_x, created_at_y, created_atのいずれかを使用
            created_at_col = None
            if "created_at" in user_data.columns:
                created_at_col = "created_at"
            elif "created_at_x" in user_data.columns:
                created_at_col = "created_at_x"
            elif "created_at_y" in user_data.columns:
                created_at_col = "created_at_y"

            if created_at_col is None:
                logger.error(
                    f"created_atカラムが見つかりません。利用可能なカラム: {list(user_data.columns)}"
                )
                return None

            logger.info(f"使用する時間カラム: {created_at_col}")

            # 時間データを確実にdatetime型に変換してからソート
            user_data_copy = user_data.copy()

            # 既にdatetime型でない場合は変換
            if not pd.api.types.is_datetime64_any_dtype(user_data_copy[created_at_col]):
                # UTC時間として解釈してから日本時間に変換
                user_data_copy[created_at_col] = pd.to_datetime(
                    user_data_copy[created_at_col], errors="coerce", utc=True
                )
                user_data_copy[created_at_col] = user_data_copy[
                    created_at_col
                ].dt.tz_convert("Asia/Tokyo")

            user_data_sorted = user_data_copy.sort_values(created_at_col)

            # 時間データを確実にdatetimeオブジェクトとして使用
            timestamps = user_data_sorted[created_at_col].tolist()

            logger.info(f"ユーザー {user_id} の時間データ: {len(timestamps)}件")

            timeseries_data = {
                "user_id": user_id,
                "timestamps": timestamps,
                "scores": user_data_sorted["score"].tolist(),
                "total_misses": user_data_sorted["total_miss"].tolist()
                if "total_miss" in user_data_sorted.columns
                else [],
            }

            logger.info(
                f"ユーザー時系列データ取得完了: {user_id}, {len(timestamps)}データポイント"
            )
            return timeseries_data

        except Exception as e:
            logger.error(f"ユーザー時系列データ取得エラー: {user_id}, {e}")
            return None

    def get_user_performance_summary(self, user_id: str) -> Dict[str, Any]:
        """
        ユーザーのパフォーマンスサマリーを取得

        Args:
            user_id: ユーザーID

        Returns:
            パフォーマンスサマリーの辞書
        """
        try:
            stats = self.get_user_stats(user_id)
            timeseries = self.get_user_timeseries(user_id)

            if not stats:
                return {
                    "status": "error",
                    "message": f"ユーザー {user_id} のデータが見つかりません",
                }

            summary = {
                "status": "success",
                "user_id": user_id,
                "stats": stats,
                "timeseries": timeseries,
                "performance_level": "普通",
                "recommendations": [],
                "timestamp": get_jst_time().isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"ユーザーパフォーマンスサマリー取得エラー: {user_id}, {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": get_jst_time().isoformat(),
            }

    def _get_cached_data(self) -> pd.DataFrame:
        """処理済みデータを取得"""
        return self.data_processor.get_processed_data()

    def get_users_performance_comparison(self, user_ids: List[str]) -> Dict[str, Any]:
        """
        複数ユーザーのパフォーマンス比較

        Args:
            user_ids: 比較するユーザーIDのリスト

        Returns:
            比較結果の辞書
        """
        try:
            comparison_data = []

            for user_id in user_ids:
                stats = self.get_user_stats(user_id)
                if stats:
                    comparison_data.append(stats)

            if not comparison_data:
                return {
                    "status": "error",
                    "message": "比較可能なユーザーデータがありません",
                }

            # 統計計算
            avg_scores = [user["avg_score"] for user in comparison_data]
            max_scores = [user["max_score"] for user in comparison_data]

            comparison_result = {
                "status": "success",
                "users": comparison_data,
                "comparison_stats": {
                    "highest_avg_score": max(avg_scores),
                    "lowest_avg_score": min(avg_scores),
                    "highest_max_score": max(max_scores),
                    "average_performance": np.mean(avg_scores),
                    "performance_std": np.std(avg_scores),
                },
                "timestamp": get_jst_time().isoformat(),
            }

            logger.info(f"ユーザー比較完了: {len(comparison_data)}人")
            return comparison_result

        except Exception as e:
            logger.error(f"ユーザー比較エラー: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": get_jst_time().isoformat(),
            }
