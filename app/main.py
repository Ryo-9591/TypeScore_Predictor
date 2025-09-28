import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class TypeScoreAnalyzer:
    def __init__(self):
        self.m_user = pd.read_csv("data/m_user.csv")
        self.t_miss = pd.read_csv("data/t_miss.csv")
        self.t_score = pd.read_csv("data/t_score.csv")
        self.models = {}
        self.model_performance = {}

        # データの前処理
        self.preprocess_data()

        # モデルの訓練
        self.train_models()

    def preprocess_data(self):
        """データの前処理"""
        # ユーザー名のマッピング
        self.user_mapping = dict(zip(self.m_user["user_id"], self.m_user["username"]))

        # スコアデータにユーザー名を追加
        self.t_score["username"] = self.t_score["user_id"].map(self.user_mapping)

        # ミスタイプデータをユーザー別に集計
        self.miss_summary = (
            self.t_miss.groupby("user_id")
            .agg({"miss_count": ["sum", "mean", "std", "count"]})
            .round(2)
        )
        self.miss_summary.columns = [
            "total_misses",
            "avg_misses",
            "std_misses",
            "miss_types",
        ]
        self.miss_summary = self.miss_summary.reset_index()

        # スコアデータとミスタイプデータを結合
        self.merged_data = self.t_score.merge(
            self.miss_summary, on="user_id", how="left"
        ).fillna(0)

        # 特徴量の作成
        self.merged_data["miss_rate"] = (
            self.merged_data["total_misses"] / self.merged_data["typing_count"]
        )
        self.merged_data["miss_rate"] = self.merged_data["miss_rate"].fillna(0)

        # 難易度と言語のラベル
        self.merged_data["difficulty_label"] = self.merged_data["diff_id"].map(
            {1: "Easy", 2: "Normal", 3: "Hard"}
        )
        self.merged_data["language_label"] = self.merged_data["lang_id"].map(
            {1: "Japanese", 2: "English"}
        )

    def train_models(self):
        """各モード（難易度×言語）ごとにモデルを訓練"""
        # 特徴量の定義
        feature_columns = [
            "diff_id",
            "lang_id",
            "accuracy",
            "typing_count",
            "total_misses",
            "avg_misses",
            "miss_rate",
        ]

        for diff_id in [1, 2, 3]:
            for lang_id in [1, 2]:
                mode_key = f"diff_{diff_id}_lang_{lang_id}"

                # 該当モードのデータを抽出
                mode_data = self.merged_data[
                    (self.merged_data["diff_id"] == diff_id)
                    & (self.merged_data["lang_id"] == lang_id)
                ].copy()

                if len(mode_data) < 10:  # データが少なすぎる場合はスキップ
                    continue

                X = mode_data[feature_columns]
                y = mode_data["score"]

                # 訓練・テスト分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # モデルの訓練
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # 予測と評価
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.models[mode_key] = model
                self.model_performance[mode_key] = {
                    "mse": mse,
                    "r2": r2,
                    "data_size": len(mode_data),
                    "test_size": len(X_test),
                }

    def get_user_predictions(self, user_id, diff_id, lang_id):
        """特定ユーザーのスコア予測"""
        mode_key = f"diff_{diff_id}_lang_{lang_id}"

        if mode_key not in self.models:
            return None

        # ユーザーの最新データを取得
        user_data = self.merged_data[
            (self.merged_data["user_id"] == user_id)
            & (self.merged_data["diff_id"] == diff_id)
            & (self.merged_data["lang_id"] == lang_id)
        ]

        if len(user_data) == 0:
            return None

        # 最新のレコードを使用
        latest_record = user_data.iloc[-1]
        feature_columns = [
            "diff_id",
            "lang_id",
            "accuracy",
            "typing_count",
            "total_misses",
            "avg_misses",
            "miss_rate",
        ]

        X = latest_record[feature_columns].values.reshape(1, -1)
        prediction = self.models[mode_key].predict(X)[0]

        return {
            "predicted_score": prediction,
            "actual_score": latest_record["score"],
            "username": latest_record["username"],
            "difficulty": latest_record["difficulty_label"],
            "language": latest_record["language_label"],
        }


# アプリケーションの初期化
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
analyzer = TypeScoreAnalyzer()

# レイアウトの定義
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "TypeScore Predictor Dashboard",
                            className="text-center mb-4",
                        ),
                        html.Hr(),
                    ]
                )
            ]
        ),
        # ユーザー選択
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("ユーザー選択"),
                        dcc.Dropdown(
                            id="user-dropdown",
                            options=[
                                {"label": username, "value": user_id}
                                for user_id, username in analyzer.user_mapping.items()
                            ],
                            value=list(analyzer.user_mapping.keys())[0],
                            className="mb-3",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3("モード選択"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("難易度"),
                                        dcc.Dropdown(
                                            id="difficulty-dropdown",
                                            options=[
                                                {"label": "Easy", "value": 1},
                                                {"label": "Normal", "value": 2},
                                                {"label": "Hard", "value": 3},
                                            ],
                                            value=2,
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("言語"),
                                        dcc.Dropdown(
                                            id="language-dropdown",
                                            options=[
                                                {"label": "Japanese", "value": 1},
                                                {"label": "English", "value": 2},
                                            ],
                                            value=1,
                                        ),
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        # 予測結果表示
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("スコア予測結果"),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            id="prediction-result",
                                            className="text-center",
                                        )
                                    ]
                                )
                            ]
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # グラフ表示エリア
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="score-correlation-plot")], width=6),
                dbc.Col([dcc.Graph(id="model-performance-plot")], width=6),
            ],
            className="mb-4",
        ),
        dbc.Row([dbc.Col([dcc.Graph(id="user-comparison-plot")], width=12)]),
    ],
    fluid=True,
)


# コールバック関数
@app.callback(
    [
        Output("prediction-result", "children"),
        Output("score-correlation-plot", "figure"),
        Output("model-performance-plot", "figure"),
        Output("user-comparison-plot", "figure"),
    ],
    [
        Input("user-dropdown", "value"),
        Input("difficulty-dropdown", "value"),
        Input("language-dropdown", "value"),
    ],
)
def update_dashboard(selected_user, selected_difficulty, selected_language):
    # 予測結果の取得
    prediction_result = analyzer.get_user_predictions(
        selected_user, selected_difficulty, selected_language
    )

    if prediction_result:
        prediction_text = f"""
        ユーザー: {prediction_result["username"]} | 
        モード: {prediction_result["difficulty"]} - {prediction_result["language"]} | 
        予測スコア: {prediction_result["predicted_score"]:.0f} | 
        実際のスコア: {prediction_result["actual_score"]:.0f}
        """
    else:
        prediction_text = "該当モードのデータが不足しています"

    # ミスタイプとスコアの相関プロット
    correlation_data = analyzer.merged_data[
        (analyzer.merged_data["diff_id"] == selected_difficulty)
        & (analyzer.merged_data["lang_id"] == selected_language)
    ]

    correlation_fig = px.scatter(
        correlation_data,
        x="total_misses",
        y="score",
        color="username",
        title=f"ミスタイプ数とスコアの相関 (難易度{selected_difficulty} - 言語{selected_language})",
        labels={"total_misses": "総ミスタイプ数", "score": "スコア"},
    )

    # モデル性能プロット
    performance_data = []
    for mode_key, perf in analyzer.model_performance.items():
        diff_id = int(mode_key.split("_")[1])
        lang_id = int(mode_key.split("_")[3])
        performance_data.append(
            {
                "mode": f"難易度{diff_id} - 言語{lang_id}",
                "R²": perf["r2"],
                "MSE": perf["mse"],
                "データ数": perf["data_size"],
            }
        )

    performance_df = pd.DataFrame(performance_data)
    performance_fig = px.bar(
        performance_df,
        x="mode",
        y="R²",
        title="モデル性能 (R²スコア)",
        labels={"mode": "モード", "R²": "R²スコア"},
    )

    # ユーザー比較プロット
    user_comparison_data = (
        analyzer.merged_data[
            (analyzer.merged_data["diff_id"] == selected_difficulty)
            & (analyzer.merged_data["lang_id"] == selected_language)
        ]
        .groupby("username")
        .agg(
            {
                "score": ["mean", "std", "count"],
                "accuracy": "mean",
                "total_misses": "mean",
            }
        )
        .round(2)
    )

    user_comparison_data.columns = [
        "avg_score",
        "std_score",
        "count",
        "avg_accuracy",
        "avg_misses",
    ]
    user_comparison_data = user_comparison_data.reset_index()

    user_comparison_fig = px.bar(
        user_comparison_data,
        x="username",
        y="avg_score",
        error_y="std_score",
        title=f"ユーザー別平均スコア比較 (難易度{selected_difficulty} - 言語{selected_language})",
        labels={"username": "ユーザー名", "avg_score": "平均スコア"},
    )

    return prediction_text, correlation_fig, performance_fig, user_comparison_fig


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
