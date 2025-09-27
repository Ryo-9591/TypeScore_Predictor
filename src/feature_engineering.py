"""
特徴量エンジニアリングモジュール
短期予測モデルに必要な、過去のパフォーマンスを示す特徴量を作成する
"""

import pandas as pd
import numpy as np
from typing import Tuple
from config import FEATURE_CONFIG


def sort_data_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """データをuser_idとタイムスタンプでソート"""
    print("データを時系列順にソート中...")

    # タイムスタンプ列を特定
    timestamp_col = next(
        (col for col in ["created_at", "updated_at"] if col in df.columns), None
    )
    if not timestamp_col:
        date_cols = [
            col for col in df.columns if "date" in col.lower() or "time" in col.lower()
        ]
        timestamp_col = date_cols[0] if date_cols else None

    if not timestamp_col:
        raise ValueError(f"タイムスタンプ列が見つかりません: {list(df.columns)}")

    df_sorted = df.sort_values(["user_id", timestamp_col]).reset_index(drop=True)
    print(f"ソート完了: {df_sorted.shape[0]}行")
    return df_sorted


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """ターゲット変数（次のセッションのスコア）を作成"""
    print("ターゲット変数を作成中...")

    df_with_target = df.copy()
    df_with_target["target_score"] = df_with_target.groupby("user_id")["score"].shift(
        -1
    )
    df_with_target = df_with_target.dropna(subset=["target_score"])

    print(f"ターゲット変数作成完了: {df.shape[0]}行 → {df_with_target.shape[0]}行")
    return df_with_target


def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """過去のパフォーマンス特徴量を作成"""
    print("過去のパフォーマンス特徴量を作成中...")

    df_with_features = df.copy().reset_index(drop=True)
    grouped = df_with_features.groupby("user_id")

    # スコア関連特徴量（インデックスを明示的にリセット）
    df_with_features["prev_score"] = grouped["score"].shift(1).reset_index(drop=True)

    rolling = grouped["score"].rolling(window=3, min_periods=1)
    df_with_features["avg_score_3"] = rolling.mean().shift(1).reset_index(drop=True)
    df_with_features["score_std_3"] = rolling.std().shift(1).reset_index(drop=True)
    df_with_features["max_score_3"] = rolling.max().shift(1).reset_index(drop=True)
    df_with_features["min_score_3"] = rolling.min().shift(1).reset_index(drop=True)

    # ミスタイプ関連特徴量
    if "total_miss" in df_with_features.columns:
        df_with_features["avg_miss_3"] = (
            grouped["total_miss"]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(drop=True)
        )
    else:
        df_with_features["avg_miss_3"] = 0

    print("特徴量作成完了")
    return df_with_features


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """特徴量とターゲットを準備"""
    print("特徴量とターゲットを準備中...")

    # 除外する列
    exclude_columns = {
        "user_id",
        "created_at_x",
        "updated_at_x",
        "created_at_y",
        "updated_at_y",
        "target_score",
        "score",
        "score_id",
        "username",
        "email",
        "password",
        "is_superuser",
        "is_staff",
        "is_active",
        "date_joined",
        "permission",
        "del_flg",
    }

    # 特徴量とターゲットを作成
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    X = df[feature_columns].copy()
    y = df["target_score"].copy()

    # 欠損値処理
    time_series_features = FEATURE_CONFIG["time_series_features"]
    for feature in time_series_features:
        if feature in X.columns:
            X[feature] = X[feature].fillna(0)

    if "rank_id" in X.columns:
        X["rank_id"] = X["rank_id"].fillna(1)

    # 欠損値を含む行を除外
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    print(f"特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}")
    return X, y


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    特徴量エンジニアリングのメイン関数

    Args:
        df: 準備されたデータフレーム

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 特徴量Xとターゲットy
    """
    # 1. データのソート
    df_sorted = sort_data_by_timestamp(df)

    # 2. ターゲット変数の作成
    df_with_target = create_target_variable(df_sorted)

    # 3. 過去のパフォーマンス特徴量の作成
    df_with_features = create_performance_features(df_with_target)

    # 4. 特徴量とターゲットの準備
    X, y = prepare_features_and_target(df_with_features)

    print("特徴量エンジニアリングが完了しました")

    return X, y


if __name__ == "__main__":
    # テスト用のダミーデータを作成
    import numpy as np

    # ダミーデータの作成
    np.random.seed(42)
    n_users = 3
    n_sessions = 10

    data = []
    for user_id in range(1, n_users + 1):
        for session in range(n_sessions):
            data.append(
                {
                    "user_id": user_id,
                    "created_at": f"2024-01-{session + 1:02d}",
                    "score": np.random.randint(50, 100),
                    "total_miss": np.random.randint(0, 10),
                }
            )

    df_test = pd.DataFrame(data)
    df_test["created_at"] = pd.to_datetime(df_test["created_at"])

    # 特徴量エンジニアリングのテスト
    X, y = engineer_features(df_test)

    print("\n特徴量の基本統計:")
    print(X.describe())
    print("\nターゲットの基本統計:")
    print(y.describe())
