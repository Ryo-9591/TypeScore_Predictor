import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core import DataProcessor, FeatureEngineer, ModelTrainer
from app.utils.common import get_logger
from app.ui.components.charts import PredictionChart, FeatureImportanceChart

logger = get_logger(__name__)


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
            scatter_fig = PredictionChart.create_scatter_plot(y_test, y_pred, metrics)

            # 特徴量重要度分析
            importance_fig = FeatureImportanceChart.create_from_model(model, X.columns)

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
                "total_records": int(len(df)),
                "unique_users": int(df["user_id"].nunique()),
                "date_range": {
                    "start": str(df["created_at"].min())
                    if "created_at" in df.columns
                    else "不明",
                    "end": str(df["created_at"].max())
                    if "created_at" in df.columns
                    else "不明",
                },
                "missing_values": {
                    k: int(v) for k, v in df.isnull().sum().to_dict().items()
                },
                "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
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
                "count": int(outlier_count),
                "percentage": float((outlier_count / len(df)) * 100),
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

    def _calculate_data_range_days(self, data_quality: Dict[str, Any]) -> int:
        """データの期間（日数）を計算"""
        try:
            date_range = data_quality.get("date_range", {})
            start_date = date_range.get("start")
            end_date = date_range.get("end")

            if start_date and end_date and start_date != "不明" and end_date != "不明":
                from datetime import datetime

                start = datetime.strptime(start_date.split()[0], "%Y-%m-%d")
                end = datetime.strptime(end_date.split()[0], "%Y-%m-%d")
                return (end - start).days
            return 0
        except:
            return 0

    def _calculate_missing_data_percentage(self, data_quality: Dict[str, Any]) -> float:
        """欠損データの割合を計算"""
        try:
            missing_values = data_quality.get("missing_values", {})
            total_records = data_quality.get("total_records", 1)

            total_missing = sum(missing_values.values())
            return (total_missing / (total_records * len(missing_values))) * 100
        except:
            return 0.0

    def _calculate_outlier_percentage(self, data_quality: Dict[str, Any]) -> float:
        """外れ値の割合を計算"""
        try:
            outliers = data_quality.get("outliers", {})
            total_records = data_quality.get("total_records", 1)

            total_outliers = sum(
                outlier_info.get("count", 0) for outlier_info in outliers.values()
            )
            return float((total_outliers / total_records) * 100)
        except:
            return 0.0
