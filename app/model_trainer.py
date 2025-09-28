import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class ModelTrainer:
    """機械学習モデルの訓練を担当するクラス"""

    def __init__(self, merged_data, feature_columns):
        """
        Args:
            merged_data (pd.DataFrame): 前処理済みの結合データ
            feature_columns (list): 特徴量の列名リスト
        """
        self.merged_data = merged_data
        self.feature_columns = feature_columns
        self.models = {}
        self.model_performance = {}

    def train_models(self):
        """各モード（難易度×言語）ごとにモデルを訓練"""
        print("モデルの訓練を開始します...")

        trained_count = 0
        for diff_id in [1, 2, 3]:
            for lang_id in [1, 2]:
                mode_key = f"diff_{diff_id}_lang_{lang_id}"

                # 該当モードのデータを抽出
                mode_data = self.merged_data[
                    (self.merged_data["diff_id"] == diff_id)
                    & (self.merged_data["lang_id"] == lang_id)
                ].copy()

                if len(mode_data) < 10:  # データが少なすぎる場合はスキップ
                    print(
                        f"モード {mode_key}: データ不足のためスキップ ({len(mode_data)}件)"
                    )
                    continue

                # モデルを訓練
                success = self._train_single_model(mode_key, mode_data)
                if success:
                    trained_count += 1

        print(f"モデルの訓練が完了しました: {trained_count}個のモデルが訓練されました")
        return trained_count > 0

    def _train_single_model(self, mode_key, mode_data):
        """単一のモードに対するモデルを訓練"""
        try:
            X = mode_data[self.feature_columns]
            y = mode_data["score"]

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # モデルの訓練
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
            )
            model.fit(X_train, y_train)

            # 予測と評価
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 結果を保存
            self.models[mode_key] = model
            self.model_performance[mode_key] = {
                "mse": mse,
                "r2": r2,
                "data_size": len(mode_data),
                "test_size": len(X_test),
                "feature_importance": dict(
                    zip(self.feature_columns, model.feature_importances_)
                ),
            }

            print(f"モード {mode_key}: 訓練完了 (R²={r2:.3f}, MSE={mse:.1f})")
            return True

        except Exception as e:
            print(f"モード {mode_key} の訓練に失敗しました: {e}")
            return False

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
        X = latest_record[self.feature_columns].values.reshape(1, -1)
        prediction = self.models[mode_key].predict(X)[0]

        return {
            "predicted_score": prediction,
            "actual_score": latest_record["score"],
            "username": latest_record["username"],
            "difficulty": latest_record["difficulty_label"],
            "language": latest_record["language_label"],
            "model_performance": self.model_performance[mode_key],
        }

    def get_models(self):
        """訓練済みモデルを返す"""
        return self.models

    def get_model_performance(self):
        """モデル性能情報を返す"""
        return self.model_performance

    def get_feature_importance(self, mode_key=None):
        """特徴量重要度を返す"""
        if mode_key:
            if mode_key in self.model_performance:
                return self.model_performance[mode_key]["feature_importance"]
            return None
        else:
            return {
                mode: perf["feature_importance"]
                for mode, perf in self.model_performance.items()
            }
