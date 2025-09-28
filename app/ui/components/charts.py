from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional

from app.utils.common import get_logger

logger = get_logger(__name__)


class PredictionChart:
    """予測精度チャートのコンポーネントクラス"""

    @staticmethod
    def create_panel(
        scatter_fig: go.Figure = None, metrics: Dict[str, Any] = None
    ) -> html.Div:
        """予測精度分析パネルを作成"""
        # 図の作成またはフォールバック
        fig_prediction = scatter_fig or PredictionChart._create_empty_figure(
            "データを取得できませんでした"
        )

        return html.Div(
            [
                html.H3("予測精度分析", className="chart-title"),
                dcc.Graph(figure=fig_prediction, className="chart-graph"),
            ],
            className="chart-panel",
        )

    @staticmethod
    def _create_empty_figure(message: str) -> go.Figure:
        """空の図を作成"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#cccccc"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            autosize=True,
        )
        return fig


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

        logger.info("特徴量重要度チャートを作成しました")
        return fig

    @staticmethod
    def create_panel(
        feature_importance: Dict[str, Any], importance_fig: go.Figure = None
    ) -> html.Div:
        """特徴量重要度パネルを作成"""
        # 図の作成またはフォールバック
        if importance_fig is not None:
            fig_feature = importance_fig
        elif feature_importance:
            fig_feature = FeatureImportanceChart._create_from_dict(feature_importance)
        else:
            fig_feature = FeatureImportanceChart._create_empty_figure(
                "データを取得できませんでした"
            )

        return html.Div(
            [
                html.H3("特徴量重要度分析", className="chart-title"),
                dcc.Graph(figure=fig_feature, className="chart-graph"),
            ],
            className="chart-panel",
        )

    @staticmethod
    def _create_from_dict(feature_importance: Dict[str, Any]) -> go.Figure:
        """辞書データからグラフを作成"""
        importance_df = pd.DataFrame(
            {
                "feature": list(feature_importance.keys()),
                "importance": list(feature_importance.values()),
            }
        ).sort_values("importance", ascending=True)

        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="特徴量重要度",
            color="importance",
            color_continuous_scale="viridis",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            margin=dict(l=80, r=40, t=60, b=40),
            autosize=True,
        )
        return fig

    @staticmethod
    def _create_empty_figure(message: str) -> go.Figure:
        """空の図を作成"""
        return PredictionChart._create_empty_figure(message)


class UserPerformanceChart:
    """ユーザーパフォーマンスチャートのコンポーネントクラス"""

    @staticmethod
    def create(
        selected_user: Optional[str], user_stats: Optional[Dict[str, Any]]
    ) -> html.Div:
        """
        ユーザーパフォーマンスチャートを作成

        Args:
            selected_user: 選択されたユーザーID
            user_stats: ユーザー統計データ

        Returns:
            ユーザーパフォーマンスチャートのDivコンポーネント
        """
        return html.Div()

    @staticmethod
    def create_with_timeseries(
        timeseries_data: Dict[str, Any], selected_user: str
    ) -> html.Div:
        """
        時系列データを使用してユーザーパフォーマンスチャートを作成

        Args:
            timeseries_data: 時系列データ
            selected_user: 選択されたユーザーID

        Returns:
            ユーザーパフォーマンスチャートのDivコンポーネント
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
            xaxis_title="",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            xaxis=dict(
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                showticklabels=True,
                tickmode="linear",
                tick0=1,
                dtick=1,
            ),
            yaxis=dict(
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                showticklabels=True,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True,
        )

        return UserPerformanceChart._create_div(fig_user)

    @staticmethod
    def _create_div(fig_user: go.Figure) -> html.Div:
        """共通のDivコンポーネントを作成"""
        return html.Div(
            [
                dcc.Graph(
                    figure=fig_user,
                    style={"height": "calc(100% - 50px)", "width": "100%"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )
