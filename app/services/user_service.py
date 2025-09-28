"""
ユーザーサービス
ユーザー関連のビジネスロジックを提供
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from app.core import DataProcessor

logger = logging.getLogger(__name__)


class UserService:
    """ユーザーサービスのメインクラス"""

    def __init__(self):
        """ユーザーサービスの初期化"""
        self.data_processor = DataProcessor()
        self._cached_data = None

    def get_all_users(self) -> List[str]:
        """
        全ユーザーのリストを取得

        Returns:
            ユーザーIDのリスト
        """
        try:
            df = self._get_cached_data()
            users = sorted(
                [str(user_id) for user_id in df["user_id"].unique().tolist()]
            )
            logger.info(f"ユーザー一覧取得: {len(users)}人")
            return users
        except Exception as e:
            logger.error(f"ユーザー一覧取得エラー: {e}")
            return []

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        指定されたユーザーの統計データを取得

        Args:
            user_id: ユーザーID

        Returns:
            ユーザー統計データの辞書
        """
        try:
            df = self._get_cached_data()
            user_data = df[df["user_id"].astype(str) == user_id]

            if len(user_data) == 0:
                logger.warning(f"ユーザーが見つかりません: {user_id}")
                return None

            # 統計計算
            total_sessions = len(user_data)
            avg_score = user_data["score"].mean()
            max_score = user_data["score"].max()
            min_score = user_data["score"].min()

            # 最新スコアの取得
            if "created_at" in user_data.columns:
                latest_score = user_data.sort_values("created_at")["score"].iloc[-1]

                # 改善傾向の計算
                recent_scores = user_data.sort_values("created_at")["score"].tail(5)
                if len(recent_scores) >= 3:
                    trend = (
                        "improving"
                        if recent_scores.iloc[-1] > recent_scores.iloc[0]
                        else "declining"
                    )
                else:
                    trend = "stable"
            else:
                latest_score = user_data["score"].iloc[-1]
                trend = "stable"

            stats = {
                "user_id": user_id,
                "total_sessions": total_sessions,
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "min_score": float(min_score),
                "latest_score": float(latest_score),
                "trend": trend,
            }

            logger.info(
                f"ユーザー統計取得完了: {user_id}, セッション数={total_sessions}"
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

            # 時系列データをソート
            if "created_at" in user_data.columns:
                user_data_sorted = user_data.sort_values("created_at")
                timestamps = (
                    user_data_sorted["created_at"]
                    .dt.strftime("%Y-%m-%d %H:%M")
                    .tolist()
                )
            else:
                # created_atがない場合はインデックスを使用
                user_data_sorted = user_data.reset_index()
                timestamps = [
                    f"セッション {i + 1}" for i in range(len(user_data_sorted))
                ]

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
                "performance_level": self._calculate_performance_level(stats),
                "recommendations": self._generate_recommendations(stats),
                "timestamp": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"ユーザーパフォーマンスサマリー取得エラー: {user_id}, {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_cached_data(self) -> pd.DataFrame:
        """キャッシュされたデータを取得"""
        if self._cached_data is None:
            self._cached_data = self.data_processor.get_processed_data()
        return self._cached_data

    def _calculate_performance_level(self, stats: Dict[str, Any]) -> str:
        """パフォーマンスレベルを計算"""
        avg_score = stats["avg_score"]

        if avg_score >= 4000:
            return "優秀"
        elif avg_score >= 3000:
            return "良好"
        elif avg_score >= 2000:
            return "普通"
        else:
            return "改善必要"

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """推奨事項を生成"""
        recommendations = []

        # スコアに基づく推奨事項
        if stats["avg_score"] < 2000:
            recommendations.append("スコア向上のため、タイピング練習を継続してください")

        # トレンドに基づく推奨事項
        if stats["trend"] == "declining":
            recommendations.append(
                "最近のスコアが下降傾向です。練習方法を見直してみてください"
            )
        elif stats["trend"] == "improving":
            recommendations.append(
                "スコアが向上しています！この調子で練習を続けてください"
            )

        # セッション数に基づく推奨事項
        if stats["total_sessions"] < 10:
            recommendations.append(
                "より多くの練習セッションで精度の高い分析が可能になります"
            )

        return recommendations

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
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"ユーザー比較完了: {len(comparison_data)}人")
            return comparison_result

        except Exception as e:
            logger.error(f"ユーザー比較エラー: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
