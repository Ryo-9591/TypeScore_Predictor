import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import warnings

# カスタムモジュールのインポート
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualizer import DataVisualizer

warnings.filterwarnings("ignore")


class TypeScoreAnalyzer:
    """タイピングスコア予測システムのメインクラス"""

    def __init__(self):
        # データ読み込み
        self.data_loader = DataLoader()
        if not self.data_loader.load_data():
            raise Exception("データの読み込みに失敗しました")

        # データ前処理
        m_user, t_miss, t_score = self.data_loader.get_data()
        self.preprocessor = DataPreprocessor(m_user, t_miss, t_score)
        self.merged_data = self.preprocessor.preprocess_data()
        self.user_mapping = self.preprocessor.get_user_mapping()

        # モデル訓練
        feature_columns = self.preprocessor.get_feature_columns()
        self.model_trainer = ModelTrainer(self.merged_data, feature_columns)
        self.model_trainer.train_models()

        # 可視化
        model_performance = self.model_trainer.get_model_performance()
        self.visualizer = DataVisualizer(self.merged_data, model_performance)

    def get_user_predictions(self, user_id, diff_id, lang_id):
        """特定ユーザーのスコア予測"""
        return self.model_trainer.get_user_predictions(user_id, diff_id, lang_id)


# アプリケーションの初期化
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
analyzer = TypeScoreAnalyzer()

# レイアウトの定義
app.layout = dbc.Container(
    [
        # ヘッダー
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "TypeScore Predictor Dashboard",
                            className="text-center mb-4 text-light",
                        ),
                        html.P(
                            "タイピングスコアの予測と分析システム",
                            className="text-center text-light mb-4",
                        ),
                        html.Hr(),
                    ]
                )
            ]
        ),
        # コントロールパネル
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H4("設定", className="mb-0")),
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "ユーザー選択",
                                            className="fw-bold text-light",
                                        ),
                                        dcc.Dropdown(
                                            id="user-dropdown",
                                            options=[
                                                {"label": username, "value": user_id}
                                                for user_id, username in analyzer.user_mapping.items()
                                            ],
                                            value=list(analyzer.user_mapping.keys())[0],
                                            className="mb-3",
                                        ),
                                        html.Label(
                                            "難易度", className="fw-bold text-light"
                                        ),
                                        dcc.Dropdown(
                                            id="difficulty-dropdown",
                                            options=[
                                                {"label": "Easy", "value": 1},
                                                {"label": "Normal", "value": 2},
                                                {"label": "Hard", "value": 3},
                                            ],
                                            value=2,
                                            className="mb-3",
                                        ),
                                        html.Label(
                                            "言語", className="fw-bold text-light"
                                        ),
                                        dcc.Dropdown(
                                            id="language-dropdown",
                                            options=[
                                                {"label": "Japanese", "value": 1},
                                                {"label": "English", "value": 2},
                                            ],
                                            value=1,
                                        ),
                                    ]
                                ),
                            ],
                            className="h-100",
                        )
                    ],
                    width=4,
                ),
                # 予測結果表示
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("スコア予測結果", className="mb-0")
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="prediction-result",
                                            className="text-center p-4",
                                        )
                                    ]
                                ),
                            ],
                            className="h-100",
                        )
                    ],
                    width=8,
                ),
            ],
            className="mb-4",
        ),
        # グラフ表示エリア
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H5(
                                        "ミスタイプとスコアの相関", className="mb-0"
                                    )
                                ),
                                dbc.CardBody([dcc.Graph(id="score-correlation-plot")]),
                            ]
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5("モデル性能", className="mb-0")),
                                dbc.CardBody([dcc.Graph(id="model-performance-plot")]),
                            ]
                        )
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


# コールバック関数
@callback(
    [
        Output("prediction-result", "children"),
        Output("score-correlation-plot", "figure"),
        Output("model-performance-plot", "figure"),
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
        # 予測結果をカード形式で表示
        prediction_display = dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("ユーザー", className="text-light"),
                                        html.H4(
                                            prediction_result["username"],
                                            className="text-info",
                                        ),
                                    ]
                                )
                            ],
                            className="text-center",
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("モード", className="text-light"),
                                        html.H4(
                                            f"{prediction_result['difficulty']} - {prediction_result['language']}",
                                            className="text-warning",
                                        ),
                                    ]
                                )
                            ],
                            className="text-center",
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("予測スコア", className="text-light"),
                                        html.H4(
                                            f"{prediction_result['predicted_score']:.0f}",
                                            className="text-success",
                                        ),
                                    ]
                                )
                            ],
                            className="text-center",
                        )
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("実際のスコア", className="text-light"),
                                        html.H4(
                                            f"{prediction_result['actual_score']:.0f}",
                                            className="text-danger",
                                        ),
                                    ]
                                )
                            ],
                            className="text-center",
                        )
                    ],
                    width=3,
                ),
            ],
            className="g-3",
        )
    else:
        prediction_display = dbc.Alert(
            "該当モードのデータが不足しています",
            color="warning",
            className="text-center",
        )

    # グラフの作成（可視化モジュールを使用）
    correlation_fig = analyzer.visualizer.create_correlation_plot(
        selected_difficulty, selected_language
    )
    performance_fig = analyzer.visualizer.create_model_performance_plot()

    return prediction_display, correlation_fig, performance_fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
