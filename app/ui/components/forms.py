"""
フォームコンポーネント
ユーザー入力用のフォームコンポーネントを作成
"""

from dash import dcc, html
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UserSelector:
    """ユーザー選択コンポーネントクラス"""

    @staticmethod
    def create(users: List[str], selected_user: Optional[str] = None) -> html.Div:
        """
        ユーザー選択ドロップダウンを作成

        Args:
            users: ユーザーIDのリスト
            selected_user: 選択されたユーザーID

        Returns:
            ユーザー選択コンポーネントのDiv
        """
        user_options = [{"label": f"ユーザー {user}", "value": user} for user in users]

        return html.Div(
            [
                html.Label(
                    "ユーザー選択:",
                    style={
                        "color": "#ffffff",
                        "fontSize": "14px",
                        "marginBottom": "5px",
                    },
                ),
                dcc.Dropdown(
                    id="user-selector",
                    options=user_options,
                    value=selected_user,
                    style={
                        "backgroundColor": "#3d3d3d",
                        "color": "#ffffff",
                        "marginBottom": "20px",
                    },
                ),
            ]
        )

    @staticmethod
    def create_with_callback(app) -> html.Div:
        """
        コールバック付きのユーザー選択コンポーネントを作成

        Args:
            app: Dashアプリケーション

        Returns:
            ユーザー選択コンポーネントのDiv
        """
        return html.Div(
            [
                html.Label(
                    "ユーザー選択:",
                    style={
                        "color": "#ffffff",
                        "fontSize": "14px",
                        "marginBottom": "5px",
                    },
                ),
                dcc.Dropdown(
                    id="user-selector",
                    options=[],  # 初期化時は空
                    value=None,
                    style={
                        "backgroundColor": "#3d3d3d",
                        "color": "#ffffff",
                        "marginBottom": "20px",
                    },
                ),
            ]
        )


class PredictionForm:
    """予測フォームコンポーネントクラス"""

    @staticmethod
    def create() -> html.Div:
        """
        予測用のフォームを作成

        Returns:
            予測フォームのDivコンポーネント
        """
        return html.Div(
            [
                html.H3(
                    "スコア予測",
                    style={
                        "color": "#ffffff",
                        "marginBottom": "15px",
                        "fontSize": "18px",
                        "textAlign": "center",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "ユーザーID:",
                            style={
                                "color": "#ffffff",
                                "fontSize": "14px",
                                "marginBottom": "5px",
                            },
                        ),
                        dcc.Input(
                            id="prediction-user-id",
                            type="text",
                            placeholder="ユーザーIDを入力",
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginBottom": "10px",
                                "backgroundColor": "#3d3d3d",
                                "color": "#ffffff",
                                "border": "1px solid #555",
                                "borderRadius": "4px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "前回スコア:",
                            style={
                                "color": "#ffffff",
                                "fontSize": "14px",
                                "marginBottom": "5px",
                            },
                        ),
                        dcc.Input(
                            id="prediction-prev-score",
                            type="number",
                            placeholder="前回のスコア",
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginBottom": "10px",
                                "backgroundColor": "#3d3d3d",
                                "color": "#ffffff",
                                "border": "1px solid #555",
                                "borderRadius": "4px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "過去3回平均スコア:",
                            style={
                                "color": "#ffffff",
                                "fontSize": "14px",
                                "marginBottom": "5px",
                            },
                        ),
                        dcc.Input(
                            id="prediction-avg-score-3",
                            type="number",
                            placeholder="過去3回の平均スコア",
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginBottom": "10px",
                                "backgroundColor": "#3d3d3d",
                                "color": "#ffffff",
                                "border": "1px solid #555",
                                "borderRadius": "4px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "タイピング数:",
                            style={
                                "color": "#ffffff",
                                "fontSize": "14px",
                                "marginBottom": "5px",
                            },
                        ),
                        dcc.Input(
                            id="prediction-typing-count",
                            type="number",
                            placeholder="タイピング数",
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginBottom": "10px",
                                "backgroundColor": "#3d3d3d",
                                "color": "#ffffff",
                                "border": "1px solid #555",
                                "borderRadius": "4px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "過去3回平均ミス数:",
                            style={
                                "color": "#ffffff",
                                "fontSize": "14px",
                                "marginBottom": "5px",
                            },
                        ),
                        dcc.Input(
                            id="prediction-avg-miss-3",
                            type="number",
                            placeholder="過去3回の平均ミス数",
                            style={
                                "width": "100%",
                                "padding": "8px",
                                "marginBottom": "15px",
                                "backgroundColor": "#3d3d3d",
                                "color": "#ffffff",
                                "border": "1px solid #555",
                                "borderRadius": "4px",
                            },
                        ),
                    ]
                ),
                html.Button(
                    "予測実行",
                    id="prediction-submit",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "backgroundColor": "#007bff",
                        "color": "#ffffff",
                        "border": "none",
                        "borderRadius": "4px",
                        "fontSize": "16px",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="prediction-result",
                    style={
                        "marginTop": "15px",
                        "padding": "10px",
                        "backgroundColor": "#2d2d2d",
                        "borderRadius": "4px",
                        "border": "1px solid #444",
                    },
                ),
            ],
            style={
                "backgroundColor": "#2d2d2d",
                "borderRadius": "8px",
                "padding": "15px",
                "marginBottom": "15px",
            },
        )

    @staticmethod
    def create_result_display(prediction_result: Dict[str, Any]) -> html.Div:
        """
        予測結果の表示コンポーネントを作成

        Args:
            prediction_result: 予測結果の辞書

        Returns:
            予測結果表示のDivコンポーネント
        """
        if not prediction_result:
            return html.Div(
                "予測結果がありません",
                style={"color": "#cccccc", "textAlign": "center"},
            )

        return html.Div(
            [
                html.H4(
                    "予測結果",
                    style={
                        "color": "#ffffff",
                        "marginBottom": "10px",
                        "textAlign": "center",
                    },
                ),
                html.Div(
                    [
                        html.Span(
                            "予測スコア: ",
                            style={"color": "#888888", "fontSize": "14px"},
                        ),
                        html.Span(
                            f"{prediction_result.get('predicted_score', 0):.0f}",
                            style={
                                "color": "#4CAF50",
                                "fontSize": "18px",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span(
                            "信頼度: ",
                            style={"color": "#888888", "fontSize": "14px"},
                        ),
                        html.Span(
                            f"{prediction_result.get('confidence', 0):.1%}",
                            style={
                                "color": "#2196F3",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span(
                            "RMSE: ",
                            style={"color": "#888888", "fontSize": "14px"},
                        ),
                        html.Span(
                            f"{prediction_result.get('model_info', {}).get('rmse', 0):.1f}",
                            style={
                                "color": "#FF9800",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                ),
            ]
        )
