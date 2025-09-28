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
    def create_scatter_plot(
        y_test, y_pred, metrics: Dict[str, Any] = None
    ) -> go.Figure:
        """
        予測結果の散布図を作成

        Args:
            y_test: 実測値
            y_pred: 予測値
            metrics: 評価指標の辞書

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

        fig.update_layout(
            xaxis_title="実測スコア",
            yaxis_title="予測スコア",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            margin=dict(l=60, r=20, t=60, b=40),
            autosize=True,
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info("予測散布図を作成しました")
        return fig

    @staticmethod
    def create_user_scatter_plot(user_predictions: pd.DataFrame) -> go.Figure:
        """
        ユーザー別最新スコアの散布図を作成

        Args:
            user_predictions: ユーザー別予測結果のDataFrame

        Returns:
            PlotlyのFigureオブジェクト
        """
        logger.info("ユーザー別最新スコアの散布図を作成中...")

        fig = go.Figure()

        # 散布図の追加
        fig.add_trace(
            go.Scatter(
                x=user_predictions["actual_score"],
                y=user_predictions["predicted_score"],
                mode="markers",
                marker=dict(
                    size=12,
                    opacity=0.7,
                    color="#28a745",
                    line=dict(color="#ffffff", width=2),
                ),
                name="ユーザー別最新スコア",
                text=[
                    f"ユーザー: {uid}<br>プレイ回数: {count}回"
                    for uid, count in zip(
                        user_predictions["user_id"], user_predictions["play_count"]
                    )
                ],
                hovertemplate="<b>%{text}</b><br>実測スコア: %{x}<br>予測スコア: %{y}<extra></extra>",
            )
        )

        # 理想的な予測線（y=x）を追加
        min_val = min(
            user_predictions["actual_score"].min(),
            user_predictions["predicted_score"].min(),
        )
        max_val = max(
            user_predictions["actual_score"].max(),
            user_predictions["predicted_score"].max(),
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#dc3545", width=3, dash="dash"),
                name="理想的な予測線",
            )
        )

        # 相関係数を計算して表示
        correlation = user_predictions["actual_score"].corr(
            user_predictions["predicted_score"]
        )

        fig.update_layout(
            xaxis_title="実測スコア（最新）",
            yaxis_title="予測スコア",
            title=f"ユーザー別最新スコア予測精度 (相関: {correlation:.3f})",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            margin=dict(l=60, r=20, t=80, b=40),
            autosize=True,
            xaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            yaxis=dict(gridcolor="#444", linecolor="#666", tickcolor="#666"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info(
            f"ユーザー別最新スコア散布図を作成しました: {len(user_predictions)}ユーザー"
        )
        return fig

    @staticmethod
    def create_panel(
        scatter_fig: go.Figure = None, metrics: Dict[str, Any] = None
    ) -> html.Div:
        """予測精度分析パネルを作成"""
        if scatter_fig is None:
            return html.Div(
                [
                    html.H3("予測精度分析", className="chart-title"),
                    html.Div(
                        "データを取得できませんでした",
                        style={
                            "color": "#cccccc",
                            "textAlign": "center",
                            "padding": "50px",
                            "fontSize": "16px",
                        },
                    ),
                ],
                className="chart-panel",
            )

        return html.Div(
            [
                html.H3("予測精度分析", className="chart-title"),
                dcc.Graph(
                    figure=scatter_fig,
                    className="chart-graph",
                    config={"displayModeBar": True, "responsive": True},
                ),
            ],
            className="chart-panel",
        )


class FeatureImportanceChart:
    """特徴量重要度チャートのコンポーネントクラス"""

    @staticmethod
    def create_from_model(model, feature_names, top_n: int = 10) -> go.Figure:
        """
        モデルから特徴量重要度チャートを作成

        Args:
            model: 学習済みモデル
            feature_names: 特徴量名のリスト
            top_n: 表示する上位特徴量数

        Returns:
            PlotlyのFigureオブジェクト
        """
        logger.info(f"モデルから特徴量重要度チャートを作成中（上位{top_n}個）...")

        # 特徴量重要度の取得
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)

        # 上位特徴量を選択
        top_features = importance_df.tail(top_n)

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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            margin=dict(l=60, r=20, t=50, b=30),
            autosize=True,
            xaxis=dict(
                title_text="重要度",
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                title_font=dict(color="#ffffff", size=12),
            ),
            yaxis=dict(
                title_text="特徴量",
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                title_font=dict(color="#ffffff", size=12),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        )

        logger.info("特徴量重要度チャートを作成しました")
        return fig

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
            return html.Div(
                [
                    html.H3("特徴量重要度分析", className="chart-title"),
                    html.Div(
                        "データを取得できませんでした",
                        style={
                            "color": "#cccccc",
                            "textAlign": "center",
                            "padding": "50px",
                            "fontSize": "16px",
                        },
                    ),
                ],
                className="chart-panel",
            )

        return html.Div(
            [
                html.H3("特徴量重要度分析", className="chart-title"),
                dcc.Graph(
                    figure=fig_feature,
                    className="chart-graph",
                    config={"displayModeBar": True, "responsive": True},
                ),
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
            margin=dict(l=60, r=20, t=50, b=30),
            autosize=True,
        )
        return fig


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
            # 時間データの型を確認して適切に処理
            timestamps = timeseries_data["timestamps"]
            scores = timeseries_data["scores"]

            logger.info(f"グラフ作成: {len(timestamps)}個のデータポイント")

            fig_user.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=scores,
                    mode="lines+markers",
                    name=f"ユーザー {selected_user}",
                    line=dict(color="#007bff", width=3),
                    marker=dict(size=6, color="#007bff"),
                )
            )

        # ダークテーマを適用
        fig_user.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=10),
            title_font=dict(color="#ffffff", size=14),
            xaxis=dict(
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                showticklabels=True,
                tickangle=45,
                tickformat="%m/%d %H:%M",
                type="date",
                tickmode="array",
                tickvals=timestamps if timestamps else [],
                ticktext=[ts.strftime("%m/%d %H:%M") for ts in timestamps]
                if timestamps
                else [],
            ),
            yaxis=dict(
                gridcolor="#444",
                linecolor="#666",
                tickcolor="#666",
                showticklabels=True,
            ),
            margin=dict(l=50, r=20, t=40, b=50),
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
                    style={"height": "100%", "width": "100%"},
                ),
            ],
            style={"height": "100%"},
        )
