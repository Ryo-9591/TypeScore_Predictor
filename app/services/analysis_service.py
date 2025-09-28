"""
分析サービス
データ分析と可視化に関するビジネスロジックを提供
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import plotly.graph_objects as go

from app.core import DataProcessor, FeatureEngineer, ModelTrainer

logger = logging.getLogger(__name__)


class AnalysisService:
    """分析サービスのメインクラス"""

    def __init__(self):
        """分析サービスの初期化"""
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self._cached_analysis = None

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        完全な分析を実行

        Returns:
            分析結果の辞書
        """
        logger.info("完全分析を開始...")
        start_time = datetime.now()

        try:
            # データ準備
            df = self.data_processor.get_processed_data()

            # 特徴量エンジニアリング
            X, y = self.feature_engineer.create_features(df)

            # モデル学習
            model, metrics = self.model_trainer.train_model(X, y)

            # 予測結果の可視化
            y_pred = model.predict(X.iloc[-int(len(X) * 0.2) :])  # テストデータの予測
            y_test = y.iloc[-int(len(y) * 0.2) :]
            scatter_fig = self.model_trainer.create_prediction_plot(y_test, y_pred)

            # 特徴量重要度分析
            importance_fig = self._create_feature_importance_chart(model, X.columns)

            # データ情報
            data_info = self.data_processor.get_data_info()

            # 評価結果
            evaluation = self.model_trainer.evaluate_performance()

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # 分析結果をまとめる
            analysis_result = {
                "status": "completed",
                "execution_time": execution_time,
                "metrics": {
                    "test_rmse": metrics["test_rmse"],
                    "test_mae": metrics["test_mae"],
                    "target_mae": self.model_trainer.config["target_mae"],
                    "mae_diff": metrics["test_mae"]
                    - self.model_trainer.config["target_mae"],
                    "achievement_status": "達成"
                    if evaluation["target_achieved"]
                    else "未達成",
                },
                "data_info": data_info,
                "feature_importance": dict(zip(X.columns, model.feature_importances_)),
                "scatter_fig": scatter_fig.to_dict() if scatter_fig else None,
                "importance_fig": importance_fig.to_dict() if importance_fig else None,
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat(),
            }

            # キャッシュに保存
            self._cached_analysis = analysis_result

            logger.info(f"完全分析完了 - 実行時間: {execution_time:.2f}秒")
            return analysis_result

        except Exception as e:
            logger.error(f"完全分析エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

    def get_cached_analysis(self) -> Optional[Dict[str, Any]]:
        """
        キャッシュされた分析結果を取得

        Returns:
            分析結果の辞書
        """
        return self._cached_analysis

    def create_feature_importance_chart(self, model, feature_names) -> go.Figure:
        """
        特徴量重要度チャートを作成

        Args:
            model: 学習済みモデル
            feature_names: 特徴量名のリスト

        Returns:
            特徴量重要度チャートのFigureオブジェクト
        """
        logger.info("特徴量重要度チャートを作成中...")

        # 特徴量重要度の取得
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)

        # 上位10個の特徴量を選択
        top_features = importance_df.tail(10)

        # ダークテーマに適したカラーパレット
        dark_colors = [
            "#007bff",
            "#28a745",
            "#ffc107",
            "#dc3545",
            "#17a2b8",
            "#6f42c1",
            "#e83e8c",
            "#17a2b8",
            "#6c757d",
            "#20c997",
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_features["importance"],
                    y=top_features["feature"],
                    orientation="h",
                    marker=dict(
                        color=dark_colors[: len(top_features)],
                        line=dict(color="#ffffff", width=1),
                    ),
                    text=[f"{imp:.3f}" for imp in top_features["importance"]],
                    textposition="auto",
                    textfont=dict(color="#ffffff", size=10),
                )
            ]
        )

        # ダークテーマを適用
        fig.update_layout(
            title="特徴量重要度（上位10個）",
            height=max(400, len(top_features) * 40),
            width=800,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=16),
            xaxis=dict(
                title_text="重要度",
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                title_font=dict(color="#ffffff", size=14),
            ),
            yaxis=dict(
                title_text="特徴量",
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                title_font=dict(color="#ffffff", size=14),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info("特徴量重要度チャートを作成しました")
        return fig

    def _create_feature_importance_chart(self, model, feature_names) -> go.Figure:
        """内部用の特徴量重要度チャート作成メソッド"""
        return self.create_feature_importance_chart(model, feature_names)

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        データ品質を分析

        Returns:
            データ品質分析結果の辞書
        """
        try:
            df = self.data_processor.get_processed_data()

            # 基本統計
            quality_analysis = {
                "total_records": len(df),
                "unique_users": df["user_id"].nunique(),
                "date_range": {
                    "start": str(df["created_at"].min())
                    if "created_at" in df.columns
                    else "不明",
                    "end": str(df["created_at"].max())
                    if "created_at" in df.columns
                    else "不明",
                },
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict(),
                "outliers": self._detect_outliers(df),
                "score_distribution": {
                    "mean": float(df["score"].mean()),
                    "std": float(df["score"].std()),
                    "min": float(df["score"].min()),
                    "max": float(df["score"].max()),
                    "median": float(df["score"].median()),
                },
            }

            logger.info("データ品質分析完了")
            return quality_analysis

        except Exception as e:
            logger.error(f"データ品質分析エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """外れ値の検出"""
        outliers = {}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            outliers[col] = {
                "count": outlier_count,
                "percentage": (outlier_count / len(df)) * 100,
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
            }

        return outliers

    def generate_insights(self) -> Dict[str, Any]:
        """
        分析結果から洞察を生成

        Returns:
            洞察の辞書
        """
        try:
            analysis = self.get_cached_analysis()
            if not analysis:
                return {
                    "status": "error",
                    "message": "分析結果がありません。先に分析を実行してください。",
                }

            insights = {
                "model_performance": {
                    "mae": analysis["metrics"]["test_mae"],
                    "target_achieved": analysis["metrics"]["achievement_status"]
                    == "達成",
                    "performance_level": self._assess_performance_level(
                        analysis["metrics"]["test_mae"]
                    ),
                },
                "data_insights": {
                    "sample_size": analysis["data_info"]["total_samples"],
                    "user_diversity": analysis["data_info"]["unique_users"],
                    "feature_count": analysis["data_info"]["feature_count"],
                },
                "feature_insights": self._analyze_feature_patterns(
                    analysis["feature_importance"]
                ),
                "recommendations": self._generate_recommendations(analysis),
            }

            logger.info("洞察生成完了")
            return insights

        except Exception as e:
            logger.error(f"洞察生成エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def _assess_performance_level(self, mae: float) -> str:
        """パフォーマンスレベルを評価"""
        if mae <= 100:
            return "優秀"
        elif mae <= 200:
            return "良好"
        elif mae <= 300:
            return "普通"
        else:
            return "改善必要"

    def _analyze_feature_patterns(
        self, feature_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """特徴量パターンを分析"""
        if not feature_importance:
            return {}

        # 特徴量をカテゴリ別に分類
        miss_features = {
            k: v for k, v in feature_importance.items() if "miss" in k.lower()
        }
        score_features = {
            k: v for k, v in feature_importance.items() if "score" in k.lower()
        }

        return {
            "most_important": max(feature_importance, key=feature_importance.get),
            "miss_features": miss_features,
            "score_features": score_features,
            "feature_diversity": len(feature_importance),
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """推奨事項を生成"""
        recommendations = []

        # モデル性能に基づく推奨事項
        if analysis["metrics"]["achievement_status"] == "未達成":
            recommendations.append(
                "モデルの精度向上のため、より多くのデータ収集を検討してください"
            )

        # 特徴量に基づく推奨事項
        feature_importance = analysis["feature_importance"]
        if feature_importance:
            most_important = max(feature_importance, key=feature_importance.get)
            recommendations.append(
                f"最も重要な特徴量 '{most_important}' を重点的に分析してください"
            )

        # データサイズに基づく推奨事項
        if analysis["data_info"]["total_samples"] < 1000:
            recommendations.append(
                "より多くのデータでモデルの安定性を向上させることができます"
            )

        return recommendations
