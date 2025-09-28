"""
チャートコンポーネント
各種グラフ・チャートの表示コンポーネントを作成
"""

from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PredictionChart:
    """予測精度チャートのコンポーネントクラス"""

    @staticmethod
    def create(
        y_test: pd.Series, y_pred: np.ndarray, metrics: Dict[str, float]
    ) -> go.Figure:
        """
        予測スコアと実測スコアの散布図を作成

        Args:
            y_test: 実測値
            y_pred: 予測値
            metrics: 評価指標

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
        title_text = f"予測スコア vs 実測スコア<br>RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}"
        fig.update_layout(
            title=title_text,
            xaxis_title="実測スコア",
            yaxis_title="予測スコア",
            height=400,
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

    @staticmethod
    def create_panel(
        scatter_fig: go.Figure = None, metrics: Dict[str, Any] = None
    ) -> html.Div:
        """
        予測精度分析パネルを作成

        Args:
            scatter_fig: 散布図のFigureオブジェクト
            metrics: 評価指標

        Returns:
            予測精度パネルのDivコンポーネント
        """
        if scatter_fig is not None:
            fig_prediction = scatter_fig
        else:
            # フォールバック: ダミーデータで散布図を作成
            fig_prediction = go.Figure()
            np.random.seed(42)
            actual_scores = np.random.normal(4000, 1500, 100)
            predicted_scores = actual_scores + np.random.normal(
                0, metrics.get("test_rmse", 200) if metrics else 200, 100
            )

            fig_prediction.add_trace(
                go.Scatter(
                    x=actual_scores,
                    y=predicted_scores,
                    mode="markers",
                    name="予測 vs 実測",
                    marker=dict(color="blue", size=6, opacity=0.6),
                )
            )

            # 完璧な予測線
            min_val = min(actual_scores.min(), predicted_scores.min())
            max_val = max(actual_scores.max(), predicted_scores.max())
            fig_prediction.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="完璧な予測",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_prediction.update_layout(
                title="予測スコア vs 実測スコア",
                xaxis_title="実測スコア",
                yaxis_title="予測スコア",
                height=400,
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#444"),
                yaxis=dict(gridcolor="#444"),
            )

        return html.Div(
            [
                html.H3(
                    "予測精度分析",
                    style={
                        "color": "#ffffff",
                        "marginBottom": "15px",
                        "fontSize": "18px",
                        "textAlign": "center",
                    },
                ),
                dcc.Graph(figure=fig_prediction, style={"height": "calc(100% - 60px)"}),
            ]
        )


class FeatureImportanceChart:
    """特徴量重要度チャートのコンポーネントクラス"""

    @staticmethod
    def create(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        特徴量重要度を棒グラフで可視化

        Args:
            importance_df: 特徴量重要度のデータフレーム
            top_n: 表示する上位特徴量数

        Returns:
            PlotlyのFigureオブジェクト
        """
        logger.info(f"特徴量重要度の棒グラフを作成中（上位{top_n}個）...")

        top_features = importance_df.head(top_n)

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
        title_text = f"特徴量重要度（上位{top_n}個）"
        fig.update_layout(
            title=title_text,
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

    @staticmethod
    def create_panel(
        feature_importance: Dict[str, Any], importance_fig: go.Figure = None
    ) -> html.Div:
        """
        特徴量重要度パネルを作成

        Args:
            feature_importance: 特徴量重要度の辞書
            importance_fig: 重要度グラフのFigureオブジェクト

        Returns:
            特徴量重要度パネルのDivコンポーネント
        """
        if importance_fig is not None:
            fig_feature = importance_fig
        elif feature_importance:
            # フォールバック: 辞書データからグラフを作成
            importance_df = pd.DataFrame(
                {
                    "feature": list(feature_importance.keys()),
                    "importance": list(feature_importance.values()),
                }
            ).sort_values("importance", ascending=True)

            fig_feature = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="特徴量重要度",
                color="importance",
                color_continuous_scale="viridis",
            )
            fig_feature.update_layout(
                height=400,
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff", size=12),
                title_font=dict(color="#ffffff", size=16),
            )
        else:
            fig_feature = go.Figure()

        return html.Div(
            [
                html.H3(
                    "特徴量重要度分析",
                    style={
                        "color": "#ffffff",
                        "marginBottom": "15px",
                        "fontSize": "18px",
                        "textAlign": "center",
                    },
                ),
                dcc.Graph(figure=fig_feature, style={"height": "calc(100% - 60px)"}),
            ]
        )


class UserPerformanceChart:
    """ユーザーパフォーマンスチャートのコンポーネントクラス"""

    @staticmethod
    def create(
        selected_user: Optional[str], user_stats: Optional[Dict[str, Any]]
    ) -> go.Figure:
        """
        ユーザーパフォーマンスチャートを作成

        Args:
            selected_user: 選択されたユーザーID
            user_stats: ユーザー統計データ

        Returns:
            ユーザーパフォーマンスチャートのFigureオブジェクト
        """
        fig_user = go.Figure()

        if selected_user and user_stats:
            # 時系列データがある場合は実際のデータを使用
            # ここでは簡易的に統計データからチャートを作成
            fig_user.add_trace(
                go.Scatter(
                    x=["過去", "現在"],
                    y=[user_stats["avg_score"], user_stats["latest_score"]],
                    mode="lines+markers",
                    name=f"ユーザー {selected_user}",
                    line=dict(color="#007bff", width=3),
                    marker=dict(size=8, color="#007bff"),
                )
            )

        # ダークテーマを適用
        fig_user.update_layout(
            title="ユーザーパフォーマンス",
            xaxis_title="日時",
            yaxis_title="スコア",
            width=800,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=16),
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        return fig_user

    @staticmethod
    def create_with_timeseries(
        timeseries_data: Dict[str, Any], selected_user: str
    ) -> go.Figure:
        """
        時系列データを使用してユーザーパフォーマンスチャートを作成

        Args:
            timeseries_data: 時系列データ
            selected_user: 選択されたユーザーID

        Returns:
            ユーザーパフォーマンスチャートのFigureオブジェクト
        """
        fig_user = go.Figure()

        if timeseries_data:
            fig_user.add_trace(
                go.Scatter(
                    x=timeseries_data["timestamps"],
                    y=timeseries_data["scores"],
                    mode="lines+markers",
                    name=f"ユーザー {selected_user}",
                    line=dict(color="#007bff", width=3),
                    marker=dict(size=6, color="#007bff"),
                )
            )

        # ダークテーマを適用
        fig_user.update_layout(
            title="ユーザーパフォーマンス",
            xaxis_title="日時",
            yaxis_title="スコア",
            width=800,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=16),
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        return fig_user
