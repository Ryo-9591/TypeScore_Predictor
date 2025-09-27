"""
モデル学習と評価モジュール
XGBoost回帰モデルを学習し、要件の評価指標に基づいて性能を評価する
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from typing import Tuple, Dict
import os
from config import MODEL_CONFIG, CV_CONFIG, TARGET_ACCURACY, OUTPUT_DIR, OUTPUT_FILES


def encode_categorical_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    カテゴリカル特徴量をエンコードする

    Args:
        X: 特徴量データフレーム

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: エンコードされた特徴量とエンコーダー
    """
    print("カテゴリカル特徴量をエンコード中...")

    X_encoded = X.copy()
    encoders = {}

    # user_idが含まれている場合、Label Encodingを適用
    if "user_id" in X_encoded.columns:
        le_user = LabelEncoder()
        X_encoded["user_id"] = le_user.fit_transform(X_encoded["user_id"])
        encoders["user_id"] = le_user
        print(f"user_idをエンコード: {len(le_user.classes_)}個のユニークユーザー")

    # その他のカテゴリカル特徴量があれば同様に処理
    categorical_columns = X_encoded.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        if col != "user_id":  # user_idは既に処理済み
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            print(f"{col}をエンコード")

    return X_encoded, encoders


def perform_time_series_split(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    時系列クロスバリデーションを実行する

    Args:
        X: 特徴量データ
        y: ターゲットデータ
        n_splits: 分割数

    Returns:
        Tuple[np.ndarray, np.ndarray]: 学習用とテスト用のインデックス
    """
    print(f"時系列クロスバリデーションを実行中（{n_splits}分割）...")

    tscv = TimeSeriesSplit(n_splits=CV_CONFIG["n_splits"])
    splits = list(tscv.split(X))

    # 最後の分割を使用（最新のデータをテストセットとして使用）
    train_idx, test_idx = splits[-1]

    # データリークを防ぐため、学習データの最後の20%を除外
    train_size = int(len(train_idx) * 0.8)
    train_idx = train_idx[:train_size]

    print(f"学習データ: {len(train_idx)}サンプル")
    print(f"テストデータ: {len(test_idx)}サンプル")
    print("データリーク防止のため、学習データの最後20%を除外しました")

    return train_idx, test_idx


def train_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    XGBoostモデルを学習し、評価する

    Args:
        X_train: 学習用特徴量
        y_train: 学習用ターゲット
        X_test: テスト用特徴量
        y_test: テスト用ターゲット

    Returns:
        Tuple[xgb.XGBRegressor, Dict[str, float]]: 学習済みモデルと評価指標
    """
    print("XGBoostモデルを学習中...")

    # XGBoost Regressorの初期化
    model = xgb.XGBRegressor(**MODEL_CONFIG)

    # モデルの学習（早期停止付き、より厳格な設定）
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,  # より早い停止
        verbose=False,
    )

    # 予測の実行
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価指標の計算
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
    }

    print("モデル学習完了")
    print(f"学習データ RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    print(f"テストデータ RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

    return model, metrics


def create_prediction_scatter_plot(
    y_test: pd.Series, y_pred: np.ndarray, metrics: Dict[str, float]
) -> None:
    """
    予測スコアと実測スコアの散布図を作成する

    Args:
        y_test: 実測値
        y_pred: 予測値
        metrics: 評価指標
    """
    print("予測結果の散布図を作成中...")

    # 散布図の作成
    fig = go.Figure()

    # 散布図の追加
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(size=8, opacity=0.6, color="blue"),
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
            line=dict(dash="dash", color="red"),
            name="理想的な予測線",
        )
    )

    # レイアウトの設定
    fig.update_layout(
        title=f"予測スコア vs 実測スコア<br>RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}",
        xaxis_title="実測スコア",
        yaxis_title="予測スコア",
        width=800,
        height=600,
    )

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # HTMLファイルとして保存
    output_file = OUTPUT_DIR / OUTPUT_FILES["scatter_plot"]
    fig.write_html(output_file)
    print(f"散布図を {output_file} に保存しました")


def evaluate_model_performance(metrics: Dict[str, float]) -> None:
    """
    モデルの性能を評価し、要件との比較を行う

    Args:
        metrics: 評価指標
    """
    print("\n=== モデル性能評価 ===")
    print(f"テストデータ RMSE: {metrics['test_rmse']:.2f}")
    print(f"テストデータ MAE: {metrics['test_mae']:.2f}")

    # 精度目標（±10点以内）との比較
    if metrics["test_mae"] <= TARGET_ACCURACY:
        print(
            f"✅ MAE ({metrics['test_mae']:.2f}) は目標精度 ({TARGET_ACCURACY}) 以内です"
        )
    else:
        print(
            f"❌ MAE ({metrics['test_mae']:.2f}) は目標精度 ({TARGET_ACCURACY}) を超えています"
        )

    print("\n=== 評価指標の解釈 ===")
    print(f"RMSE: 予測誤差の標準偏差。{metrics['test_rmse']:.2f}点の誤差")
    print(f"MAE: 平均絶対誤差。{metrics['test_mae']:.2f}点の誤差")


def train_and_evaluate_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    モデル学習と評価のメイン関数

    Args:
        X: 特徴量データ
        y: ターゲットデータ

    Returns:
        Tuple[xgb.XGBRegressor, Dict[str, float]]: 学習済みモデルと評価指標
    """
    # 1. カテゴリカル特徴量のエンコード
    X_encoded, encoders = encode_categorical_features(X)

    # 2. 時系列クロスバリデーション
    train_idx, test_idx = perform_time_series_split(X_encoded, y)

    # 3. データの分割
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 4. XGBoostモデルの学習と評価
    model, metrics = train_xgboost_model(X_train, y_train, X_test, y_test)

    # 5. 予測結果の可視化
    y_pred = model.predict(X_test)
    create_prediction_scatter_plot(y_test, y_pred, metrics)

    # 6. 性能評価
    evaluate_model_performance(metrics)

    return model, metrics


if __name__ == "__main__":
    # テスト用のダミーデータを作成
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    # ダミーデータの作成
    X_test = pd.DataFrame(
        {
            "user_id": np.random.randint(1, 11, n_samples),
            "score": np.random.randint(50, 100, n_samples),
            "total_miss": np.random.randint(0, 10, n_samples),
            "prev_score": np.random.randint(50, 100, n_samples),
            "avg_score_3": np.random.uniform(50, 100, n_samples),
            "avg_miss_3": np.random.uniform(0, 10, n_samples),
        }
    )

    y_test = np.random.randint(50, 100, n_samples)

    # モデル学習と評価のテスト
    model, metrics = train_and_evaluate_model(X_test, pd.Series(y_test))

    print("\n特徴量重要度:")
    feature_importance = pd.DataFrame(
        {"feature": X_test.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(feature_importance)
