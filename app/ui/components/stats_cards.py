"""
統計カードコンポーネント
ダッシュボード用の統計表示カードを作成
"""

from dash import html
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StatsCard:
    """統計カードのコンポーネントクラス"""

    @staticmethod
    def create(
        title: str,
        value: str,
        description: str,
        color: str = "#007bff",
        size: str = "normal",  # "small", "normal", "large"
    ) -> html.Div:
        """
        統計カードを作成

        Args:
            title: カードのタイトル
            value: 表示する値
            description: 説明文
            color: 値の色
            size: カードサイズ ("small", "normal", "large")

        Returns:
            DashのDivコンポーネント
        """
        # サイズに応じたスタイル設定
        size_config = {
            "small": {
                "padding": "15px",
                "titleFontSize": "12px",
                "valueFontSize": "22px",
                "descFontSize": "11px",
            },
            "normal": {
                "padding": "20px",
                "titleFontSize": "14px",
                "valueFontSize": "28px",
                "descFontSize": "12px",
            },
            "large": {
                "padding": "25px",
                "titleFontSize": "16px",
                "valueFontSize": "32px",
                "descFontSize": "13px",
            },
        }

        config = size_config.get(size, size_config["normal"])

        return html.Div(
            [
                html.H3(
                    title,
                    style={
                        "color": "#ffffff",
                        "fontSize": config["titleFontSize"],
                        "margin": "0 0 5px 0",
                        "fontWeight": "500",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                    },
                ),
                html.H2(
                    value,
                    style={
                        "color": color,
                        "fontSize": config["valueFontSize"],
                        "margin": "0 0 8px 0",
                        "fontWeight": "700",
                    },
                ),
                html.P(
                    description,
                    style={
                        "color": "#cccccc",
                        "fontSize": config["descFontSize"],
                        "margin": "0",
                        "opacity": "0.8",
                    },
                ),
            ],
            className="stats-card",
            style={
                "backgroundColor": "#2d2d2d",
                "border": "1px solid #444",
                "borderRadius": "12px",
                "padding": config["padding"],
                "textAlign": "center",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
                # ホバー効果とトランジションを削除
            },
        )

    @staticmethod
    def create_user_stats_card(user_stats: Dict[str, Any], user_id: str) -> html.Div:
        """
        ユーザー統計カードを作成

        Args:
            user_stats: ユーザー統計データ
            user_id: ユーザーID

        Returns:
            ユーザー統計カード
        """
        if not user_stats:
            return html.Div(
                [
                    html.P(
                        "データを取得できませんでした",
                        style={
                            "color": "#cccccc",
                            "textAlign": "center",
                            "padding": "20px",
                        },
                    )
                ]
            )

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "総セッション数",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['total_sessions']}",
                                    style={
                                        "color": "#ffffff",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "平均スコア",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['avg_score']:.0f}",
                                    style={
                                        "color": "#ffffff",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "最高スコア",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['max_score']:.0f}",
                                    style={
                                        "color": "#4CAF50",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "marginBottom": "10px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "最低スコア",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['min_score']:.0f}",
                                    style={
                                        "color": "#FF9800",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "最新スコア",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['latest_score']:.0f}",
                                    style={
                                        "color": "#2196F3",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "改善傾向",
                                    style={
                                        "color": "#888888",
                                        "fontSize": "11px",
                                        "display": "block",
                                    },
                                ),
                                html.Span(
                                    f"{user_stats['trend']}",
                                    style={
                                        "color": "#4CAF50"
                                        if user_stats["trend"] == "improving"
                                        else "#FF5722"
                                        if user_stats["trend"] == "declining"
                                        else "#FFC107",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "padding": "8px",
                                "backgroundColor": "#3d3d3d",
                                "borderRadius": "5px",
                                "margin": "3px",
                                "flex": "1",
                                "minWidth": "100px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "width": "100%",
                "marginBottom": "15px",
            },
        )


class StatsGrid:
    """統計グリッドのコンポーネントクラス"""

    @staticmethod
    def create_global_stats_grid(
        metrics: Dict[str, Any],
        data_info: Dict[str, Any],
        analysis_data: Dict[str, Any],
    ) -> List[html.Div]:
        """
        グローバル統計グリッドを作成

        Args:
            metrics: 評価指標
            data_info: データ情報
            analysis_data: 分析データ

        Returns:
            統計カードのリスト
        """
        execution_time_str = f"{analysis_data['execution_time']:.2f}秒"

        return [
            # RMSE
            StatsCard.create(
                "RMSE", f"{metrics['test_rmse']:.1f}", "予測誤差の標準偏差", "#2E8B57"
            ),
            # MAE
            StatsCard.create(
                "MAE",
                f"{metrics['test_mae']:.1f}",
                "平均絶対誤差",
                "#4169E1",
            ),
            # 特徴量数
            StatsCard.create(
                "特徴量数", str(data_info["feature_count"]), "入力変数", "#9370DB"
            ),
            # 実行時間
            StatsCard.create("実行時間", execution_time_str, "処理時間", "#ff9ff3"),
        ]
