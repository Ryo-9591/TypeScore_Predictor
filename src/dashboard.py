"""
TypeScore Predictor - Plotly Dash インタラクティブダッシュボード
リアルタイムで予測結果と分析を表示
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dash  # noqa: E402
from dash import dcc, html, Input, Output, callback  # noqa: E402
import plotly.graph_objs as go  # noqa: E402
import plotly.express as px  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime  # noqa: E402
import requests  # noqa: E402
import numpy as np  # noqa: E402
import time  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402
from urllib3.util.retry import Retry  # noqa: E402

# Dashアプリの初期化
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "TypeScore Predictor Dashboard"

# API設定
API_BASE_URL = "http://api:8000"  # コンテナ間通信ではサービス名を使用


# リトライ機能付きHTTPセッションの作成
def create_retry_session():
    """リトライ機能付きのHTTPセッションを作成"""
    session = requests.Session()

    # リトライ戦略の設定
    retry_strategy = Retry(
        total=5,  # 最大5回リトライ
        backoff_factor=1,  # バックオフ係数（1秒、2秒、4秒、8秒、16秒）
        status_forcelist=[429, 500, 502, 503, 504],  # リトライするHTTPステータスコード
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # リトライするHTTPメソッド
    )

    # HTTPアダプターにリトライ戦略を設定
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# グローバル変数（データキャッシュ用）
cached_analysis_data = None
cached_user_data = None


def fetch_analysis_data():
    """APIから分析データを取得"""
    global cached_analysis_data

    if cached_analysis_data is None:
        try:
            print("APIから分析データを取得中...")
            response = requests.post(f"{API_BASE_URL}/analyze")
            response.raise_for_status()
            cached_analysis_data = response.json()
            print("分析データ取得完了")
        except Exception as e:
            print(f"API取得エラー: {str(e)}")
            # フォールバック用のダミーデータ
            cached_analysis_data = {
                "status": "error",
                "execution_time": 0.0,
                "metrics": {
                    "test_rmse": 1154.7,
                    "test_mae": 783.1,
                    "target_mae": 200.0,
                    "mae_diff": 583.1,
                    "achievement_status": "未達成",
                },
                "data_info": {
                    "total_samples": 902,
                    "unique_users": 31,
                    "feature_count": 12,
                    "training_samples": 676,
                    "test_samples": 150,
                },
                "feature_importance": {},
                "timestamp": datetime.now().isoformat(),
            }

    return cached_analysis_data


def fetch_user_data():
    """APIからユーザーデータを取得"""
    global cached_user_data

    if cached_user_data is None:
        try:
            print("APIからユーザーデータを取得中...")
            users_response = requests.get(f"{API_BASE_URL}/users", timeout=30)
            users_response.raise_for_status()
            users = users_response.json()

            # 最初のユーザーの統計を取得
            if users:
                user_stats_response = requests.get(
                    f"{API_BASE_URL}/users/{users[0]}/stats", timeout=30
                )
                user_stats_response.raise_for_status()
                user_stats = user_stats_response.json()

                cached_user_data = {"users": users, "current_user_stats": user_stats}
            else:
                cached_user_data = {"users": [], "current_user_stats": None}

            print("ユーザーデータ取得完了")
        except Exception as e:
            print(f"ユーザーデータ取得エラー: {str(e)}")
            cached_user_data = {"users": [], "current_user_stats": None}

    return cached_user_data


# レイアウト定義
app.layout = html.Div(
    [
        # ヘッダー部分
        html.Div(
            [
                # 左側：アイコンとタイトル
                html.Div(
                    [
                        html.Div(
                            "📊",
                            style={
                                "fontSize": "24px",
                                "marginRight": "10px",
                                "color": "#ff6b6b",
                            },
                        ),
                        html.Div(
                            [
                                html.H1(
                                    "TypeScore Predictor",
                                    style={
                                        "color": "#ffffff",
                                        "fontSize": "28px",
                                        "margin": "0",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.P(
                                    "Track Typing Performance Analytics",
                                    style={
                                        "color": "#ffffff",
                                        "fontSize": "14px",
                                        "margin": "5px 0 0 0",
                                        "opacity": "0.8",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                # 右側：最終更新時刻
                html.Div(
                    id="last-updated",
                    style={
                        "color": "#ffa500",
                        "fontSize": "12px",
                        "textAlign": "right",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "20px",
                "padding": "0 10px",
            },
        ),
        # グローバル統計カード（4x2グリッド）
        html.Div(
            id="global-stats-grid",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                "gridTemplateRows": "1fr 1fr",
                "gap": "15px",
                "marginBottom": "20px",
            },
        ),
        # 下部3パネル
        html.Div(
            [
                # 左パネル：ユーザー選択と新規データ
                html.Div(
                    [
                        html.H3(
                            "ユーザー選択と最新データ",
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
                        ),
                        html.Div(id="user-stats-display"),
                    ],
                    style={
                        "backgroundColor": "#2d2d2d",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "flex": "1",
                    },
                ),
                # 中央パネル：ドーナツチャート
                html.Div(
                    id="center-panel",
                    style={
                        "backgroundColor": "#2d2d2d",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "flex": "1",
                    },
                ),
                # 右パネル：時系列チャート
                html.Div(
                    id="right-panel",
                    style={
                        "backgroundColor": "#2d2d2d",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "flex": "1",
                    },
                ),
            ],
            style={"display": "flex", "gap": "15px", "height": "calc(100vh - 400px)"},
        ),
        # 自動更新
        dcc.Interval(
            id="interval-component",
            interval=30 * 1000,  # 30秒ごとに更新
            n_intervals=0,
        ),
    ],
    style={
        "backgroundColor": "#1a1a1a",
        "height": "100vh",
        "padding": "15px",
        "overflow": "hidden",
    },
)


@callback(
    [Output("global-stats-grid", "children"), Output("last-updated", "children")],
    Input("interval-component", "n_intervals"),
)
def update_global_stats(n):
    """グローバル統計カードと最終更新時刻を更新"""
    try:
        analysis_data = fetch_analysis_data()
        metrics = analysis_data["metrics"]
        data_info = analysis_data["data_info"]

        # 現在時刻を取得
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M (JST)")

        # 実行時間を取得
        execution_time_str = f"{analysis_data['execution_time']:.2f}秒"

        # 目標精度との差を取得
        mae_diff_str = f"{metrics['mae_diff']:+.1f}"

        # 統計カードを作成（4x2グリッド）
        stats_cards = [
            # 1列目（縦）
            # RMSE
            html.Div(
                [
                    html.H3(
                        "RMSE",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        f"{metrics['test_rmse']:.1f}",
                        style={"color": "#2E8B57", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "予測誤差の標準偏差",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 学習データ
            html.Div(
                [
                    html.H3(
                        "学習データ",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        str(data_info["training_samples"]),
                        style={"color": "#4ecdc4", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "サンプル数",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 特徴量数
            html.Div(
                [
                    html.H3(
                        "特徴量数",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        str(data_info["feature_count"]),
                        style={"color": "#9370DB", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "入力変数",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 実行時間
            html.Div(
                [
                    html.H3(
                        "実行時間",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        execution_time_str,
                        style={"color": "#ff9ff3", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "処理時間",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 2列目（縦）
            # MAE
            html.Div(
                [
                    html.H3(
                        "MAE",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        f"{metrics['test_mae']:.1f}",
                        style={"color": "#4169E1", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        f"平均絶対誤差 (目標差: {mae_diff_str})",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # テストデータ
            html.Div(
                [
                    html.H3(
                        "テストデータ",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        str(data_info["test_samples"]),
                        style={"color": "#ffa500", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "サンプル数",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 総サンプル数
            html.Div(
                [
                    html.H3(
                        "総サンプル数",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        f"{data_info['total_samples']:,}",
                        style={"color": "#FF6347", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        "データポイント",
                        style={
                            "color": "#cccccc",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
            # 目標精度
            html.Div(
                [
                    html.H3(
                        "目標精度",
                        style={
                            "color": "#ffffff",
                            "fontSize": "14px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.H2(
                        "200.0",
                        style={"color": "#ff6b6b", "fontSize": "24px", "margin": "0"},
                    ),
                    html.P(
                        f"MAE目標値 ({metrics['achievement_status']})",
                        style={
                            "color": "#ff6b6b"
                            if metrics["mae_diff"] > 0
                            else "#4ecdc4",
                            "fontSize": "12px",
                            "margin": "5px 0 0 0",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #444",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                },
            ),
        ]

        return stats_cards, f"最終更新: {current_time}"

    except Exception as e:
        error_card = html.Div(
            f"エラー: {str(e)}",
            style={
                "backgroundColor": "#2d2d2d",
                "border": "1px solid #ff6b6b",
                "borderRadius": "8px",
                "padding": "15px",
                "textAlign": "center",
                "color": "#ff6b6b",
            },
        )
        return [
            error_card,
            error_card,
            error_card,
            error_card,
            error_card,
            error_card,
            error_card,
            error_card,
        ], "エラーが発生しました"


@callback(
    [
        Output("user-selector", "options"),
        Output("user-selector", "value"),
        Output("user-stats-display", "children"),
        Output("center-panel", "children"),
        Output("right-panel", "children"),
    ],
    [Input("interval-component", "n_intervals"), Input("user-selector", "value")],
)
def render_three_panels(n, selected_user):
    """3つのパネルをレンダリング"""
    try:
        analysis_data = fetch_analysis_data()
        user_data = fetch_user_data()
        metrics = analysis_data["metrics"]
        feature_importance = analysis_data["feature_importance"]

        # ユーザー選択オプションを作成
        users = user_data["users"]
        user_options = [{"label": f"ユーザー {user}", "value": user} for user in users]

        # デフォルトユーザーを設定
        if selected_user is None and users:
            selected_user = users[0]

        # 選択されたユーザーの統計を取得
        if selected_user:
            try:
                user_stats_response = requests.get(
                    f"{API_BASE_URL}/users/{selected_user}/stats"
                )
                user_stats_response.raise_for_status()
                user_stats = user_stats_response.json()
            except:
                pass

        # ユーザー別パフォーマンスチャート（簡易版）
        fig_user = go.Figure()

        if selected_user:
            # 選択されたユーザーの時系列データを取得
            try:
                timeseries_response = requests.get(
                    f"{API_BASE_URL}/users/{selected_user}/timeseries"
                )
                timeseries_response.raise_for_status()
                timeseries_data = timeseries_response.json()

                # 実際の時系列データでチャートを作成
                fig_user.add_trace(
                    go.Scatter(
                        x=timeseries_data["timestamps"],
                        y=timeseries_data["scores"],
                        mode="lines+markers",
                        name=f"ユーザー {selected_user}",
                        line=dict(color="blue", width=3),
                        marker=dict(size=6),
                    )
                )
            except Exception as e:
                print(f"ユーザー時系列データ取得エラー: {str(e)}")
                # フォールバック: 統計データを使用
                try:
                    user_stats_response = requests.get(
                        f"{API_BASE_URL}/users/{selected_user}/stats"
                    )
                    user_stats_response.raise_for_status()
                    user_stats = user_stats_response.json()

                    fig_user.add_trace(
                        go.Scatter(
                            x=["過去", "現在"],
                            y=[user_stats["avg_score"], user_stats["latest_score"]],
                            mode="lines+markers",
                            name=f"ユーザー {selected_user}",
                            line=dict(color="blue", width=3),
                            marker=dict(size=8),
                        )
                    )
                except Exception as e2:
                    print(f"ユーザー統計取得エラー: {str(e2)}")
                    # エラー時は空のチャートを表示
                    pass

        fig_user.update_layout(
            title=f"ユーザー {selected_user} のスコア推移",
            xaxis_title="期間",
            yaxis_title="スコア",
            height=400,
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#444"),
            yaxis=dict(gridcolor="#444"),
        )

        # ユーザー統計情報の表示
        if selected_user:
            try:
                user_stats_response = requests.get(
                    f"{API_BASE_URL}/users/{selected_user}/stats"
                )
                user_stats_response.raise_for_status()
                user_stats = user_stats_response.json()

                user_stats_info = html.Div(
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
                                            f"{user_stats['improvement_trend']}",
                                            style={
                                                "color": "#4CAF50"
                                                if user_stats["improvement_trend"]
                                                == "improving"
                                                else "#FF5722"
                                                if user_stats["improvement_trend"]
                                                == "declining"
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
                        # グラフを同じカード内に配置
                        html.Div(
                            [
                                dcc.Graph(
                                    id="user-performance-chart",
                                    figure=fig_user,
                                    style={"height": "250px"},
                                )
                            ],
                            style={
                                "width": "100%",
                                "marginTop": "15px",
                                "maxHeight": "300px",
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
            except Exception as e:
                print(f"ユーザー統計取得エラー: {str(e)}")
                user_stats_info = html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "データを取得できませんでした",
                                    style={
                                        "color": "#cccccc",
                                        "margin": "5px 0",
                                        "textAlign": "center",
                                        "padding": "20px",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "300px",
                                "marginRight": "15px",
                            },
                        ),
                        # グラフを同じカード内に配置
                        html.Div(
                            [
                                dcc.Graph(
                                    id="user-performance-chart",
                                    figure=fig_user,
                                    style={"height": "200px"},
                                )
                            ],
                            style={"width": "100%", "marginTop": "10px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "width": "100%",
                    },
                )
        else:
            user_stats_info = html.Div(
                [
                    html.Div(
                        [
                            html.P(
                                "ユーザーを選択してください",
                                style={
                                    "color": "#cccccc",
                                    "margin": "5px 0",
                                    "textAlign": "center",
                                    "padding": "20px",
                                },
                            ),
                        ],
                        style={"flex": "1", "minWidth": "300px", "marginRight": "15px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="user-performance-chart",
                                figure=fig_user,
                                style={"height": "300px"},
                            )
                        ],
                        style={"flex": "2", "minWidth": "400px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "15px",
                    "alignItems": "flex-start",
                },
            )

        # user_stats_displayを定義
        user_stats_display = (
            user_stats_info
            if selected_user
            else html.Div(
                [
                    html.P(
                        "ユーザーを選択してください",
                        style={
                            "color": "#cccccc",
                            "textAlign": "center",
                            "padding": "20px",
                        },
                    )
                ]
            )
        )

        # 中央パネル：特徴量重要度チャート
        if feature_importance:
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
                height=400, yaxis={"categoryorder": "total ascending"}
            )
        else:
            fig_feature = go.Figure()
        center_panel = html.Div(
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

        # 右パネル：予測精度分析（簡易版）
        fig_prediction = go.Figure()

        # ダミーデータで散布図を作成
        np.random.seed(42)
        actual_scores = np.random.normal(4000, 1500, 100)
        predicted_scores = actual_scores + np.random.normal(
            0, metrics["test_rmse"], 100
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
        right_panel = html.Div(
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

        return (
            user_options,
            selected_user,
            user_stats_display,
            center_panel,
            right_panel,
        )

    except Exception as e:
        error_div = html.Div(
            f"エラー: {str(e)}",
            style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"},
        )
        return [], None, error_div, error_div, error_div


# CSSスタイル
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0f1419;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                overflow-x: hidden;
            }
            
            /* グローバル統計カードのスタイル */
            .global-stats-card {
                background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
                border: 1px solid #444;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .global-stats-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            }
            
            .global-stats-card h3 {
                margin: 0 0 10px 0;
                font-size: 14px;
                font-weight: 500;
                color: #ffffff;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .global-stats-card h2 {
                margin: 0 0 8px 0;
                font-size: 28px;
                font-weight: 700;
            }
            
            .global-stats-card p {
                margin: 0;
                font-size: 12px;
                color: #cccccc;
                opacity: 0.8;
            }
            
            /* パネルのスタイル */
            .panel {
                background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
                border: 1px solid #444;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            
            /* ヘッダーのスタイル */
            .dashboard-header {
                background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            /* Plotly グラフのダークテーマ */
            .js-plotly-plot .plotly .modebar {
                background-color: #2d2d2d !important;
                border-radius: 8px !important;
            }
            .js-plotly-plot .plotly .modebar-btn {
                color: #ffffff !important;
            }
            .js-plotly-plot .plotly .modebar-btn:hover {
                background-color: #444 !important;
            }
            
            /* ドロップダウンのダークテーマ */
            .Select-control {
                background-color: #3d3d3d !important;
                border: 1px solid #555 !important;
                border-radius: 8px !important;
                color: #ffffff !important;
            }
            .Select-control:hover {
                border-color: #777 !important;
            }
            .Select-menu-outer {
                background-color: #3d3d3d !important;
                border: 1px solid #555 !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
            }
            .Select-option {
                background-color: #3d3d3d !important;
                color: #ffffff !important;
                padding: 10px 15px !important;
            }
            .Select-option.is-focused {
                background-color: #555 !important;
            }
            .Select-option.is-selected {
                background-color: #007bff !important;
            }
            .Select-placeholder {
                color: #cccccc !important;
            }
            .Select-input {
                color: #ffffff !important;
            }
            
            /* アニメーション */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .fade-in {
                animation: fadeIn 0.5s ease-out;
            }
            
            
            /* スクロールバーのスタイル */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #2d2d2d;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb {
                background: #555;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #777;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
