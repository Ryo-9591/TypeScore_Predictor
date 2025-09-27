"""
データ準備とクリーニングモジュール
3つのCSVファイルを結合し、セッション単位の統一データセットを作成する
"""

import pandas as pd
import numpy as np
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
    df_score = pd.read_csv(DATA_DIR / SCORE_FILE)
    df_miss = pd.read_csv(DATA_DIR / MISS_FILE)
    df_user = pd.read_csv(DATA_DIR / USER_FILE)

    print(f"t_score: {df_score.shape[0]}行")
    print(f"t_miss: {df_miss.shape[0]}行")
    print(f"m_user: {df_user.shape[0]}行")

    return df_score, df_miss, df_user


def aggregate_miss_data(df_miss: pd.DataFrame) -> pd.DataFrame:
    """
    t_missをuser_idとcreated_atでグループ化し、miss_countの合計を計算する

    Args:
        df_miss: t_missのデータフレーム

    Returns:
        pd.DataFrame: 集計されたミスデータ
    """
    print("t_missデータを集計中...")

    # タイムスタンプ列の特定（created_atまたはupdated_at）
    timestamp_col = None
    for col in ["created_at", "updated_at"]:
        if col in df_miss.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(
            "タイムスタンプ列（created_atまたはupdated_at）が見つかりません"
        )

    # user_idとタイムスタンプでグループ化してmiss_countの合計を計算
    df_miss_agg = (
        df_miss.groupby(["user_id", timestamp_col])["miss_count"].sum().reset_index()
    )
    df_miss_agg.rename(columns={"miss_count": "total_miss"}, inplace=True)

    print(f"集計後のt_miss: {df_miss_agg.shape[0]}行")

    return df_miss_agg


def merge_score_and_miss(
    df_score: pd.DataFrame, df_miss_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    df_scoreとdf_missを結合する

    Args:
        df_score: スコアデータ
        df_miss_agg: 集計されたミスデータ

    Returns:
        pd.DataFrame: 結合されたデータフレーム
    """
    print("スコアデータとミスデータを結合中...")

    # タイムスタンプ列の特定
    score_timestamp_col = None
    miss_timestamp_col = None

    for col in ["created_at", "updated_at"]:
        if col in df_score.columns:
            score_timestamp_col = col
        if col in df_miss_agg.columns:
            miss_timestamp_col = col

    if score_timestamp_col is None or miss_timestamp_col is None:
        raise ValueError("タイムスタンプ列が見つかりません")

    # 結合キーを確認
    print(f"スコアデータの結合キー: user_id, {score_timestamp_col}")
    print(f"ミスデータの結合キー: user_id, {miss_timestamp_col}")

    # 結合キーが異なる場合は、ミスデータの列名をスコアデータに合わせる
    if score_timestamp_col != miss_timestamp_col:
        df_miss_agg = df_miss_agg.rename(
            columns={miss_timestamp_col: score_timestamp_col}
        )
        print(f"ミスデータの列名を変更: {miss_timestamp_col} → {score_timestamp_col}")

    # 左外部結合を実行
    df_merged = df_score.merge(
        df_miss_agg, on=["user_id", score_timestamp_col], how="left"
    )

    # total_missの欠損値を0で補完
    df_merged["total_miss"] = df_merged["total_miss"].fillna(0)

    print(f"結合後のデータ: {df_merged.shape[0]}行")
    print(f"total_missの欠損値補完: {df_merged['total_miss'].isna().sum()}個")

    return df_merged


def merge_with_user_data(
    df_merged: pd.DataFrame, df_user: pd.DataFrame
) -> pd.DataFrame:
    """
    df_mergedとm_userをuser_idで結合する

    Args:
        df_merged: スコアとミスが結合されたデータ
        df_user: ユーザー情報データ

    Returns:
        pd.DataFrame: 最終的なデータフレーム
    """
    print("ユーザー情報を結合中...")

    # user_idで結合
    df_final = df_merged.merge(df_user, on="user_id", how="left")

    # 日付型への変換
    timestamp_cols = ["created_at", "updated_at"]
    for col in timestamp_cols:
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
