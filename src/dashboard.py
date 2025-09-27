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

from src.data_preparation import prepare_data  # noqa: E402
from src.feature_engineering import engineer_features  # noqa: E402
from src.model_training import train_and_evaluate_model  # noqa: E402

# Dashアプリの初期化
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "TypeScore Predictor Dashboard"

# グローバル変数（データキャッシュ用）
cached_data = None
cached_model = None
cached_metrics = None
execution_time = None


def load_and_cache_data():
    """データとモデルを読み込んでキャッシュ"""
    global cached_data, cached_model, cached_metrics, execution_time

    if cached_data is None:
        print("データを読み込み中...")
        start_time = datetime.now()

        df_final = prepare_data()
        X, y = engineer_features(df_final)
        model, metrics = train_and_evaluate_model(X, y)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        cached_data = {"df_final": df_final, "X": X, "y": y}
        cached_model = model
        cached_metrics = metrics

        print(f"データ読み込み完了 - 実行時間: {execution_time:.2f}秒")

    return cached_data, cached_model, cached_metrics


def create_user_performance_chart(df_final):
    """ユーザー別パフォーマンス推移チャート"""

    # ユーザー選択用のドロップダウン
    users = sorted(df_final["user_id"].unique())

    fig = go.Figure()

    # 全ユーザーの平均スコアを薄い線で表示
    avg_scores = df_final.groupby("created_at_x")["score"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=avg_scores["created_at_x"],
            y=avg_scores["score"],
            mode="lines",
            name="全体平均",
            line=dict(color="lightgray", width=1, dash="dot"),
        )
    )

    return fig, users


def create_feature_importance_chart(model, feature_names):
    """特徴量重要度チャート"""
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
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

    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})

    return fig


def create_prediction_scatter(model, X, y):
    """予測 vs 実測散布図"""
    y_pred = model.predict(X)

    fig = px.scatter(
        x=y,
        y=y_pred,
        title="予測スコア vs 実測スコア",
        labels={"x": "実測スコア", "y": "予測スコア"},
    )

    # 完璧な予測線を追加
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="完璧な予測",
            line=dict(color="red", dash="dash"),
        )
    )

    return fig


def create_donut_chart(df_final):
    """スコア分布のドーナツチャート"""
    # スコア範囲で分類
    df_final_copy = df_final.copy()
    df_final_copy["score_category"] = pd.cut(
        df_final_copy["score"],
        bins=[0, 50, 70, 85, 100],
        labels=[
            "低スコア (0-50)",
            "中スコア (50-70)",
            "高スコア (70-85)",
            "最高スコア (85-100)",
        ],
    )

    category_counts = df_final_copy["score_category"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.6,  # ドーナツチャートにする
                marker_colors=["#ff6b6b", "#ffa500", "#4ecdc4", "#ff9ff3"],
                textinfo="label+value+percent",
                textfont=dict(color="white", size=12),
            )
        ]
    )

    fig.update_layout(
        title="スコア分布",
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01),
    )

    return fig


def create_time_series_chart(df_final):
    """時系列チャート（最近30日間のスコア推移）"""
    # 日付でグループ化して平均スコアを計算
    df_final_copy = df_final.copy()
    df_final_copy["date"] = pd.to_datetime(df_final_copy["created_at_x"]).dt.date

    # 最近30日間のデータを取得
    latest_date = df_final_copy["date"].max()
    thirty_days_ago = latest_date - pd.Timedelta(days=30)
    recent_data = df_final_copy[df_final_copy["date"] >= thirty_days_ago]

    daily_stats = recent_data.groupby("date").agg({"score": ["mean", "count"]}).round(1)
    daily_stats.columns = ["平均スコア", "セッション数"]
    daily_stats = daily_stats.reset_index()

    # 移動平均を計算
    daily_stats["移動平均（7日間）"] = (
        daily_stats["平均スコア"].rolling(window=7, min_periods=1).mean()
    )

    fig = go.Figure()

    # 平均スコアのバーチャート
    fig.add_trace(
        go.Bar(
            x=daily_stats["date"],
            y=daily_stats["平均スコア"],
            name="日別平均スコア",
            marker_color="#ffa500",
            opacity=0.7,
        )
    )

    # 移動平均のライン
    fig.add_trace(
        go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["移動平均（7日間）"],
            mode="lines",
            name="移動平均（7日間）",
            line=dict(color="#9b59b6", width=3),
        )
    )

    fig.update_layout(
        title="最近30日間のスコア推移",
        xaxis_title="日付",
        yaxis_title="スコア",
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#444"),
        yaxis=dict(gridcolor="#444"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    )

    return fig


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
        data, model, metrics = load_and_cache_data()
        df_final = data["df_final"]

        # 現在時刻を取得
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M (JST)")

        # 実行時間を取得
        execution_time_str = f"{execution_time:.2f}秒" if execution_time else "N/A"

        # 目標精度との差を計算
        mae_diff = metrics["test_mae"] - 200.0
        mae_diff_str = f"{mae_diff:+.1f}" if mae_diff >= 0 else f"{mae_diff:.1f}"

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
                        "676",
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
                        f"{data['X'].shape[1]}",
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
                        "150",
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
                        f"{data['X'].shape[0]:,}",
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
                        f"MAE目標値 ({'未達成' if mae_diff > 0 else '達成'})",
                        style={
                            "color": "#ff6b6b" if mae_diff > 0 else "#4ecdc4",
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
        data, model, metrics = load_and_cache_data()
        df_final = data["df_final"]

        # ユーザー選択オプションを作成
        users = sorted(df_final["user_id"].unique())
        user_options = [{"label": f"ユーザー {user}", "value": user} for user in users]

        # デフォルトユーザーを設定
        if selected_user is None and users:
            selected_user = users[0]

        # 選択されたユーザーの最新データを取得
        if selected_user:
            user_data = df_final[df_final["user_id"] == selected_user].tail(1).iloc[0]
            latest_score = user_data["score"]
            previous_score = df_final[df_final["user_id"] == selected_user].tail(2)
            if len(previous_score) > 1:
                score_change = latest_score - previous_score.iloc[0]["score"]
                score_change_text = (
                    f"{score_change:+.1f}" if not pd.isna(score_change) else "N/A"
                )
            else:
                score_change_text = "N/A"
        else:
            latest_score = 0
            score_change_text = "N/A"

        # ユーザー別パフォーマンスチャート
        fig_user, _ = create_user_performance_chart(df_final)

        # 選択されたユーザーのデータを追加
        if selected_user:
            user_data = df_final[df_final["user_id"] == selected_user].sort_values(
                "created_at_x"
            )
            fig_user.add_trace(
                go.Scatter(
                    x=user_data["created_at_x"],
                    y=user_data["score"],
                    mode="lines+markers",
                    name=f"ユーザー {selected_user}",
                    line=dict(color="blue", width=3),
                    marker=dict(size=8),
                )
            )

        fig_user.update_layout(
            title=f"ユーザー {selected_user} のスコア推移",
            xaxis_title="日付",
            yaxis_title="スコア",
            height=400,
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#444"),
            yaxis=dict(gridcolor="#444"),
        )

        user_stats_display = html.Div(
            [
                dcc.Graph(
                    id="user-performance-chart",
                    figure=fig_user,
                    style={"height": "400px"},
                ),
            ]
        )

        # 中央パネル：特徴量重要度チャート
        fig_feature = create_feature_importance_chart(model, data["X"].columns)
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

        # 右パネル：予測精度分析
        fig_prediction = create_prediction_scatter(model, data["X"], data["y"])
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
