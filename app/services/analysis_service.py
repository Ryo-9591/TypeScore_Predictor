import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import plotly.graph_objects as go

from app.core import DataProcessor, FeatureEngineer, ModelTrainer
from app.config import PREDICTION_REPORT_CONFIG
from app.utils import safe_text_log
from app.logging_config import get_logger, get_report_logger

logger = get_logger(__name__)

# 予測精度レポート用の専用ロガー
report_logger = get_report_logger()


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
            scatter_fig = self.create_prediction_plot(y_test, y_pred)

            # 特徴量重要度分析
            importance_fig = self.create_feature_importance_chart(model, X.columns)

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

            # 分析レポートの出力
            self._log_analysis_report(analysis_result, execution_time)

            return analysis_result

        except Exception as e:
            logger.error(f"完全分析エラー: {e}")
            self._log_analysis_error(
                str(e), (datetime.now() - start_time).total_seconds()
            )
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

    def create_prediction_plot(
        self, y_test: pd.Series, y_pred: np.ndarray
    ) -> go.Figure:
        """
        予測結果の散布図を作成

        Args:
            y_test: 実測値
            y_pred: 予測値

        Returns:
            PlotlyのFigureオブジェクト
        """
        logger.info("予測結果の散布図を作成中...")

        fig = go.Figure()

        # 散布図の追加
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color="#007bff",
                    line=dict(color="#ffffff", width=1),
                ),
                name="予測値 vs 実測値",
            )
        )

        # 理想的な予測線（y=x）を追加
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#dc3545", width=3, dash="dash"),
                name="理想的な予測線",
            )
        )

        # ダークテーマを適用
        metrics = self.model_trainer.metrics
        title_text = f"予測スコア vs 実測スコア<br>RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}"
        fig.update_layout(
            title=title_text,
            xaxis_title="実測スコア",
            yaxis_title="予測スコア",
            width=800,
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=16),
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info("予測散布図を作成しました")
        return fig

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

    def _log_analysis_report(
        self, analysis_result: Dict[str, Any], execution_time: float
    ):
        """分析レポートをログに出力"""
        if not PREDICTION_REPORT_CONFIG["enabled"]:
            return

        try:
            # データ品質分析
            data_quality = self.analyze_data_quality()

            # 分析レポート
            analysis_report = {
                "event_type": "full_analysis",
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "status": analysis_result["status"],
                "data_quality": data_quality,
                "model_performance": analysis_result.get("metrics", {}),
                "data_info": analysis_result.get("data_info", {}),
                "feature_importance": analysis_result.get("feature_importance", {}),
                "analysis_summary": {
                    "total_samples": int(data_quality.get("total_records", 0)),
                    "unique_users": int(data_quality.get("unique_users", 0)),
                    "data_range_days": int(
                        self._calculate_data_range_days(data_quality)
                    ),
                    "missing_data_percentage": float(
                        self._calculate_missing_data_percentage(data_quality)
                    ),
                    "outlier_percentage": float(
                        self._calculate_outlier_percentage(data_quality)
                    ),
                },
            }

            report_logger.info(safe_text_log(analysis_report, "ANALYSIS_REPORT"))

        except Exception as e:
            logger.error(f"分析レポート出力エラー: {e}")

    def _log_analysis_error(self, error_message: str, execution_time: float):
        """分析エラーレポートをログに出力"""
        error_report = {
            "event_type": "analysis_error",
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "execution_time": execution_time,
        }

        report_logger.error(safe_text_log(error_report, "ANALYSIS_ERROR_REPORT"))

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

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的な予測精度レポートを生成"""
        try:
            # 完全分析の実行
            analysis_result = self.run_full_analysis()

            if analysis_result["status"] != "completed":
                return {"status": "error", "message": "分析の実行に失敗しました"}

            # データ品質分析
            data_quality = self.analyze_data_quality()

            # 包括的レポートの生成
            comprehensive_report = {
                "report_generated_at": datetime.now().isoformat(),
                "report_type": "comprehensive_prediction_accuracy",
                "executive_summary": {
                    "model_performance": {
                        "mae": float(analysis_result["metrics"]["test_mae"]),
                        "rmse": float(analysis_result["metrics"]["test_rmse"]),
                        "target_achieved": bool(
                            analysis_result["metrics"]["test_mae"] <= 200.0
                        ),
                        "performance_level": self._assess_overall_performance(
                            analysis_result["metrics"]
                        ),
                    },
                    "data_quality_score": self._calculate_data_quality_score(
                        data_quality
                    ),
                    "recommendations": self._generate_comprehensive_recommendations(
                        analysis_result, data_quality
                    ),
                },
                "detailed_analysis": {
                    "model_metrics": analysis_result["metrics"],
                    "data_info": analysis_result["data_info"],
                    "feature_importance": analysis_result["feature_importance"],
                    "data_quality": data_quality,
                },
                "technical_details": {
                    "execution_time": float(analysis_result["execution_time"]),
                    "feature_count": int(len(analysis_result["feature_importance"])),
                    "sample_count": int(analysis_result["data_info"]["total_samples"]),
                },
            }

            # レポートをログに出力
            report_logger.info(
                safe_text_log(comprehensive_report, "COMPREHENSIVE_REPORT")
            )

            return {"status": "success", "report": comprehensive_report}

        except Exception as e:
            logger.error(f"包括的レポート生成エラー: {e}")
            return {"status": "error", "error": str(e)}

    def _assess_overall_performance(self, metrics: Dict[str, float]) -> str:
        """全体的なパフォーマンスを評価"""
        mae = metrics["test_mae"]
        rmse = metrics["test_rmse"]

        if mae <= 200 and rmse <= 300:
            return "優秀"
        elif mae <= 300 and rmse <= 500:
            return "良好"
        elif mae <= 500 and rmse <= 800:
            return "普通"
        else:
            return "改善必要"

    def _calculate_data_quality_score(self, data_quality: Dict[str, Any]) -> float:
        """データ品質スコアを計算（0-100）"""
        try:
            score = 100.0

            # 欠損データによる減点
            missing_percentage = self._calculate_missing_data_percentage(data_quality)
            score -= min(missing_percentage * 2, 30)  # 最大30点減点

            # 外れ値による減点
            outlier_percentage = self._calculate_outlier_percentage(data_quality)
            score -= min(outlier_percentage * 0.5, 20)  # 最大20点減点

            # データ量による調整
            total_records = data_quality.get("total_records", 0)
            if total_records < 100:
                score -= 20
            elif total_records < 500:
                score -= 10

            return max(score, 0.0)
        except:
            return 50.0

    def _generate_comprehensive_recommendations(
        self, analysis_result: Dict[str, Any], data_quality: Dict[str, Any]
    ) -> List[str]:
        """包括的な改善推奨事項を生成"""
        recommendations = []

        # モデル性能に基づく推奨事項
        mae = analysis_result["metrics"]["test_mae"]
        if mae > 300:
            recommendations.append("モデルパラメータの調整を検討してください")
            recommendations.append("特徴量エンジニアリングの見直しを推奨します")

        # データ品質に基づく推奨事項
        missing_percentage = self._calculate_missing_data_percentage(data_quality)
        if missing_percentage > 10:
            recommendations.append("欠損データの補完方法を改善してください")

        outlier_percentage = self._calculate_outlier_percentage(data_quality)
        if outlier_percentage > 5:
            recommendations.append("外れ値の処理方法を見直してください")

        # データ量に基づく推奨事項
        total_records = data_quality.get("total_records", 0)
        if total_records < 500:
            recommendations.append("より多くの学習データの収集を推奨します")

        if not recommendations:
            recommendations.append(
                "現在のモデル性能は良好です。継続的な監視を推奨します"
            )

        return recommendations
