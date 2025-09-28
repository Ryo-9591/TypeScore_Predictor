"""
TypeScore Predictor - メインアプリケーション
新しいアーキテクチャに基づく統合アプリケーション
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go

# 新しいアーキテクチャのインポート
from app.services import PredictionService, UserService, AnalysisService
from app.ui.components import (
    StatsCard,
    StatsGrid,
    PredictionChart,
    FeatureImportanceChart,
    UserPerformanceChart,
    UserSelector,
)
from app.ui.styles import get_layout_styles, get_css_styles
from app.config import DASHBOARD_CONFIG

# Dashアプリの初期化
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = DASHBOARD_CONFIG["title"]

# サービス層の初期化
prediction_service = PredictionService()
user_service = UserService()
analysis_service = AnalysisService()

# グローバル変数（データキャッシュ用）
cached_analysis_data = None


def load_data_and_model():
    """データとモデルを読み込んでキャッシュ"""
    global cached_analysis_data

    if cached_analysis_data is None:
        print("データとモデルの読み込み中...")
        start_time = datetime.now()

        # 分析サービスを使用してデータとモデルを読み込み
        cached_analysis_data = analysis_service.run_full_analysis()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        print(f"データとモデルの読み込み完了 - 実行時間: {execution_time:.2f}秒")

    return cached_analysis_data


def get_user_data() -> Dict[str, Any]:
    """ユーザーデータを取得"""
    users = user_service.get_all_users()
    return {"users": users}


def get_user_stats(user_id: str) -> Optional[Dict[str, Any]]:
    """指定されたユーザーの統計データを取得"""
    return user_service.get_user_stats(user_id)


def get_user_timeseries(user_id: str) -> Optional[Dict[str, Any]]:
    """指定されたユーザーの時系列データを取得"""
    return user_service.get_user_timeseries(user_id)


def create_user_performance_chart(
    selected_user: Optional[str], user_stats: Optional[Dict[str, Any]]
) -> go.Figure:
    """ユーザーパフォーマンスチャートを作成"""
    if selected_user:
        timeseries_data = get_user_timeseries(selected_user)
        if timeseries_data:
            return UserPerformanceChart.create_with_timeseries(
                timeseries_data, selected_user
            )

    return UserPerformanceChart.create(selected_user, user_stats)


def create_user_stats_display(
    selected_user: Optional[str],
    user_stats: Optional[Dict[str, Any]],
    fig_user: go.Figure,
) -> html.Div:
    """ユーザー統計表示を作成"""
    if not selected_user:
        return html.Div(
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

    if not user_stats:
        return html.Div(
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

    # ユーザー統計情報とグラフを組み合わせて表示
    user_stats_info = StatsCard.create_user_stats_card(user_stats, selected_user)

    return html.Div(
        [
            user_stats_info,
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


def create_feature_importance_panel(
    feature_importance: Dict[str, Any], importance_fig: go.Figure = None
) -> html.Div:
    """特徴量重要度パネルを作成"""
    return FeatureImportanceChart.create_panel(feature_importance, importance_fig)


def create_prediction_accuracy_panel(
    metrics: Dict[str, Any], scatter_fig: go.Figure = None
) -> html.Div:
    """予測精度分析パネルを作成"""
    return PredictionChart.create_panel(scatter_fig, metrics)


# レイアウトスタイルを取得
layout_styles = get_layout_styles()

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
            style=layout_styles["header"],
        ),
        # グローバル統計カード（4x1グリッド）
        html.Div(
            id="global-stats-grid",
            style=layout_styles["stats_grid"],
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
                        html.Div(id="user-selector-container"),
                        html.Div(id="user-stats-display"),
                    ],
                    style=layout_styles["panel"],
                ),
                # 中央パネル：特徴量重要度
                html.Div(
                    id="center-panel",
                    style=layout_styles["panel"],
                ),
                # 右パネル：予測精度分析
                html.Div(
                    id="right-panel",
                    style=layout_styles["panel"],
                ),
            ],
            style=layout_styles["panel_container"],
        ),
        # 自動更新
        dcc.Interval(
            id="interval-component",
            interval=30 * 1000,  # 30秒ごとに更新
            n_intervals=0,
        ),
    ],
    style=layout_styles["main_container"],
)


@callback(
    [Output("global-stats-grid", "children"), Output("last-updated", "children")],
    Input("interval-component", "n_intervals"),
)
def update_global_stats(n: int) -> Tuple[List[html.Div], str]:
    """グローバル統計カードと最終更新時刻を更新"""
    try:
        # データとモデルを読み込み
        analysis_data = load_data_and_model()

        if analysis_data["status"] != "completed":
            error_card = html.Div(
                f"エラー: {analysis_data.get('error', '不明なエラー')}",
                style={
                    "backgroundColor": "#2d2d2d",
                    "border": "1px solid #ff6b6b",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "textAlign": "center",
                    "color": "#ff6b6b",
                },
            )
            return [error_card] * 4, "エラーが発生しました"

        metrics = analysis_data["metrics"]
        data_info = analysis_data["data_info"]

        # 現在時刻を取得
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M (JST)")

        # 統計カードを作成
        stats_cards = StatsGrid.create_global_stats_grid(
            metrics, data_info, analysis_data
        )
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
        return [error_card] * 4, "エラーが発生しました"


@callback(
    [
        Output("user-selector-container", "children"),
        Output("user-stats-display", "children"),
        Output("center-panel", "children"),
        Output("right-panel", "children"),
    ],
    [Input("interval-component", "n_intervals")],
)
def render_panels(n: int) -> Tuple[html.Div, html.Div, html.Div, html.Div]:
    """パネルをレンダリング"""
    try:
        # データとモデルを読み込み
        analysis_data = load_data_and_model()

        if analysis_data["status"] != "completed":
            error_div = html.Div(
                f"エラー: {analysis_data.get('error', '不明なエラー')}",
                style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"},
            )
            return error_div, error_div, error_div, error_div

        # ユーザーデータを取得
        user_data = get_user_data()
        users = user_data["users"]

        # ユーザー選択コンポーネントを作成
        user_selector = UserSelector.create(users, users[0] if users else None)

        # デフォルトユーザーを設定
        selected_user = users[0] if users else None

        # 選択されたユーザーの統計を取得
        user_stats = None
        if selected_user:
            user_stats = get_user_stats(selected_user)

        # ユーザー別パフォーマンスチャートを作成
        fig_user = create_user_performance_chart(selected_user, user_stats)

        # ユーザー統計情報の表示
        user_stats_display = create_user_stats_display(
            selected_user, user_stats, fig_user
        )

        # グラフオブジェクトを取得
        scatter_fig = None
        importance_fig = None

        if "scatter_fig" in analysis_data and analysis_data["scatter_fig"]:
            scatter_fig = go.Figure(analysis_data["scatter_fig"])

        if "importance_fig" in analysis_data and analysis_data["importance_fig"]:
            importance_fig = go.Figure(analysis_data["importance_fig"])

        # 中央パネルと右パネルを作成
        center_panel = create_feature_importance_panel(
            analysis_data["feature_importance"], importance_fig
        )
        right_panel = create_prediction_accuracy_panel(
            analysis_data["metrics"], scatter_fig
        )

        return user_selector, user_stats_display, center_panel, right_panel

    except Exception as e:
        error_div = html.Div(
            f"エラー: {str(e)}",
            style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"},
        )
        return error_div, error_div, error_div, error_div


# CSSスタイルを適用
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {get_css_styles()}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run_server(
        debug=DASHBOARD_CONFIG["debug"],
        host=DASHBOARD_CONFIG["host"],
        port=DASHBOARD_CONFIG["port"],
    )
