"""
特徴量エンジニアリングモジュール
短期予測モデルに必要な、過去のパフォーマンスを示す特徴量を作成する
"""

import pandas as pd
import numpy as np
from typing import Tuple
from config import FEATURE_CONFIG


def sort_data_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    データをuser_idとcreated_atでソートし、時系列順に並べる

    Args:
        df: 入力データフレーム

    Returns:
        pd.DataFrame: ソートされたデータフレーム
    """
    print("データを時系列順にソート中...")

    # 利用可能な列を確認
    print(f"利用可能な列: {list(df.columns)}")

    # タイムスタンプ列の特定
    timestamp_col = None
    for col in ["created_at", "updated_at"]:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        # タイムスタンプ列が見つからない場合、日付関連の列を探す
        date_columns = [
            col for col in df.columns if "date" in col.lower() or "time" in col.lower()
        ]
        if date_columns:
            timestamp_col = date_columns[0]
            print(f"日付関連列を使用: {timestamp_col}")
        else:
            raise ValueError(
                f"タイムスタンプ列が見つかりません。利用可能な列: {list(df.columns)}"
            )

    print(f"使用するタイムスタンプ列: {timestamp_col}")

    # user_idとタイムスタンプでソート
    df_sorted = df.sort_values(["user_id", timestamp_col]).reset_index(drop=True)

    print(f"ソート完了: {df_sorted.shape[0]}行")

    return df_sorted


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    ターゲット変数（次のセッションのスコア）を作成する

    Args:
        df: ソートされたデータフレーム

    Returns:
        pd.DataFrame: ターゲット変数が追加されたデータフレーム
    """
    print("ターゲット変数を作成中...")

    df_with_target = df.copy()

    # ユーザーIDごとにscore列を1行上にずらした列を作成
    df_with_target["target_score"] = df_with_target.groupby("user_id")["score"].shift(
        -1
    )

    # target_scoreがNaNの最終レコードを除外
    initial_rows = df_with_target.shape[0]
    df_with_target = df_with_target.dropna(subset=["target_score"])
    final_rows = df_with_target.shape[0]

    print(f"ターゲット変数作成完了: {initial_rows}行 → {final_rows}行")

    return df_with_target


def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去のパフォーマンス特徴量を作成する

    Args:
        df: ターゲット変数が追加されたデータフレーム

    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    print("過去のパフォーマンス特徴量を作成中...")

    df_with_features = df.copy().reset_index(drop=True)

    # ユーザーIDごとにグループ化して特徴量を作成
    grouped = df_with_features.groupby("user_id")

    # 直前のセッションのスコア
    prev_score = grouped["score"].shift(1).reset_index(drop=True)
    df_with_features["prev_score"] = prev_score

    # 過去3回の平均スコア（Rolling Mean）
    avg_score_3 = (
        grouped["score"]
        .rolling(window=3, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(drop=True)
    )
    df_with_features["avg_score_3"] = avg_score_3

    # 過去3回の平均ミスタイプ数
    if "total_miss" in df_with_features.columns:
        avg_miss_3 = (
            grouped["total_miss"]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(drop=True)
        )
        df_with_features["avg_miss_3"] = avg_miss_3
        
        # ミスタイプ率の特徴量を追加
        if "typing_count" in df_with_features.columns:
            # ミスタイプ率 = ミスタイプ数 / タイピング数
            miss_rate = (
                grouped["total_miss"] / (grouped["typing_count"] + 1)  # +1でゼロ除算を回避
                .rolling(window=3, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(drop=True)
            )
            df_with_features["miss_rate_3"] = miss_rate
            
            # 直前のミスタイプ率
            prev_miss_rate = (
                grouped["total_miss"] / (grouped["typing_count"] + 1)
                .shift(1)
                .reset_index(drop=True)
            )
            df_with_features["prev_miss_rate"] = prev_miss_rate
    else:
        print("警告: total_miss列が見つかりません。ミスタイプ関連特徴量は作成されません。")
        df_with_features["avg_miss_3"] = 0
        df_with_features["miss_rate_3"] = 0
        df_with_features["prev_miss_rate"] = 0

    # その他の有用な特徴量
    # 過去のスコアの標準偏差（安定性の指標）
    score_std_3 = (
        grouped["score"]
        .rolling(window=3, min_periods=1)
        .std()
        .shift(1)
        .reset_index(drop=True)
    )
    df_with_features["score_std_3"] = score_std_3

    # 過去のスコアの最大値
    max_score_3 = (
        grouped["score"]
        .rolling(window=3, min_periods=1)
        .max()
        .shift(1)
        .reset_index(drop=True)
    )
    df_with_features["max_score_3"] = max_score_3

    # 過去のスコアの最小値
    min_score_3 = (
        grouped["score"]
        .rolling(window=3, min_periods=1)
        .min()
        .shift(1)
        .reset_index(drop=True)
    )
    df_with_features["min_score_3"] = min_score_3

    print("特徴量作成完了")

    return df_with_features


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    特徴量とターゲットを準備する

    Args:
        df: 特徴量が追加されたデータフレーム

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 特徴量Xとターゲットy
    """
    print("特徴量とターゲットを準備中...")

    # 除外する列を定義
    exclude_columns = [
        "user_id",
        "created_at_x",
        "updated_at_x",
        "created_at_y",
        "updated_at_y",
        "target_score",
        "score_id",  # ID列は予測に不要
        "username",  # 匿名化済みのため予測に不要
        "email",     # 匿名化済みのため予測に不要
        "password",  # セキュリティ上除外
        "is_superuser",  # システム管理用のため予測に不要
        "is_staff",      # システム管理用のため予測に不要
        "is_active",     # システム管理用のため予測に不要
        "date_joined",   # 登録日は予測に不要
        "permission",    # システム管理用のため予測に不要
        "del_flg",       # 削除フラグは予測に不要
    ]

    # 特徴量Xを作成（除外列を除く）
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    X = df[feature_columns].copy()

    # ターゲットyを作成
    y = df["target_score"].copy()

    # 欠損値の確認
    print(f"欠損値の確認:")
    print(f"  Xの欠損値: {X.isna().sum().sum()}")
    print(f"  yの欠損値: {y.isna().sum()}")

    # 各列の欠損値を詳細表示
    print("各列の欠損値:")
    for col in X.columns:
        missing_count = X[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count}個")

    # 欠損値を含む行を除外（時系列特徴量の欠損値は0で補完）
    initial_rows = X.shape[0]

    # 時系列特徴量の欠損値を0で補完
    time_series_features = FEATURE_CONFIG["time_series_features"] + [
        "miss_rate_3", "prev_miss_rate"
    ]
    for feature in time_series_features:
        if feature in X.columns:
            X[feature] = X[feature].fillna(0)
            print(f"  {feature}の欠損値を0で補完: {X[feature].isna().sum()}個")

    # その他の重要な特徴量の欠損値を補完
    if "rank_id" in X.columns:
        X["rank_id"] = X["rank_id"].fillna(1)  # デフォルトランクを1に設定
        print(f"  rank_idの欠損値を1で補完: {X['rank_id'].isna().sum()}個")

    # その他の欠損値を含む行を除外
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"欠損値処理後: {initial_rows}行 → {X.shape[0]}行")

    print(f"特徴量数: {X.shape[1]}")
    print(f"サンプル数: {X.shape[0]}")
    print(f"特徴量名: {list(X.columns)}")

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
