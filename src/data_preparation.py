"""
データ準備とクリーニングモジュール
3つのCSVファイルを結合し、セッション単位の統一データセットを作成する
"""

import pandas as pd
from typing import Tuple
from config import DATA_DIR, SCORE_FILE, MISS_FILE, USER_FILE


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3つのCSVファイルを読み込む

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: df_score, df_miss, df_user
    """
    print("データファイルを読み込み中...")

    # CSVファイルの読み込み
    df_score = pd.read_csv(DATA_DIR / SCORE_FILE, low_memory=False)
    df_miss = pd.read_csv(DATA_DIR / MISS_FILE, low_memory=False)
    df_user = pd.read_csv(DATA_DIR / USER_FILE, low_memory=False)

    print(f"t_score: {df_score.shape[0]}行")
    print(f"t_miss: {df_miss.shape[0]}行")
    print(f"m_user: {df_user.shape[0]}行")

    return df_score, df_miss, df_user


def aggregate_miss_data(df_miss: pd.DataFrame) -> pd.DataFrame:
    """t_missをuser_idとタイムスタンプでグループ化し、miss_countの合計を計算"""
    print("t_missデータを集計中...")

    # タイムスタンプ列を特定
    timestamp_col = next(
        (col for col in ["created_at", "updated_at"] if col in df_miss.columns), None
    )
    if not timestamp_col:
        raise ValueError("タイムスタンプ列が見つかりません")

    # 集計実行
    df_miss_agg = (
        df_miss.groupby(["user_id", timestamp_col], as_index=False)["miss_count"]
        .sum()
        .rename(columns={"miss_count": "total_miss"})
    )

    print(f"集計後のt_miss: {df_miss_agg.shape[0]}行")
    return df_miss_agg


def merge_score_and_miss(
    df_score: pd.DataFrame, df_miss_agg: pd.DataFrame
) -> pd.DataFrame:
    """df_scoreとdf_missを結合"""
    print("スコアデータとミスデータを結合中...")

    print(f"df_score columns: {list(df_score.columns)}")
    print(f"df_miss_agg columns: {list(df_miss_agg.columns)}")

    # タイムスタンプ列を特定
    score_col = next(
        (col for col in ["created_at", "updated_at"] if col in df_score.columns), None
    )
    miss_col = next(
        (col for col in ["created_at", "updated_at"] if col in df_miss_agg.columns),
        None,
    )

    print(f"score_col: {score_col}, miss_col: {miss_col}")

    if not score_col or not miss_col:
        raise ValueError(
            f"タイムスタンプ列が見つかりません - score_col: {score_col}, miss_col: {miss_col}"
        )

    # 列名を統一
    if score_col != miss_col:
        df_miss_agg = df_miss_agg.rename(columns={miss_col: score_col})

    # 結合実行
    df_merged = df_score.merge(df_miss_agg, on=["user_id", score_col], how="left")
    df_merged["total_miss"] = df_merged["total_miss"].fillna(0)

    print(f"結合後のデータ: {df_merged.shape[0]}行")
    return df_merged


def merge_with_user_data(
    df_merged: pd.DataFrame, df_user: pd.DataFrame
) -> pd.DataFrame:
    """df_mergedとm_userをuser_idで結合"""
    print("ユーザー情報を結合中...")

    df_final = df_merged.merge(df_user, on="user_id", how="left")

    # 重複するカラム名を処理（created_at_xをcreated_atに統一）
    if "created_at_x" in df_final.columns:
        df_final = df_final.rename(columns={"created_at_x": "created_at"})
    if "updated_at_x" in df_final.columns:
        df_final = df_final.rename(columns={"updated_at_x": "updated_at"})

    # 日付型への変換
    for col in ["created_at", "updated_at"]:
        if col in df_final.columns:
            df_final[col] = pd.to_datetime(df_final[col])

    print(f"最終データ: {df_final.shape[0]}行, {df_final.shape[1]}列")
    return df_final


def prepare_data() -> pd.DataFrame:
    """
    データ準備のメイン関数

    Returns:
        pd.DataFrame: 準備されたデータフレーム
    """
    # 1. データの読み込み
    df_score, df_miss, df_user = load_data()

    # 2. t_missの集計
    df_miss_agg = aggregate_miss_data(df_miss)

    # 3. スコアとミスデータの結合
    df_merged = merge_score_and_miss(df_score, df_miss_agg)

    # 4. ユーザー情報の結合
    df_final = merge_with_user_data(df_merged, df_user)

    print("データ準備が完了しました")
    return df_final


if __name__ == "__main__":
    # テスト実行
    df = prepare_data()
    print("\nデータの基本情報:")
    print(df.info())
    print("\n最初の5行:")
    print(df.head())
