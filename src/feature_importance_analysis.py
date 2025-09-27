"""
特徴量重要度分析モジュール
スコアに最も影響を与えている要因を特定し、フィードバックとシミュレーションの基盤とする
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import os
from config import OUTPUT_DIR, OUTPUT_FILES


def get_feature_importance(
    model: xgb.XGBRegressor, feature_names: List[str]
) -> pd.DataFrame:
    """学習済みXGBoostモデルから特徴量重要度を取得"""
    print("特徴量重要度を取得中...")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"特徴量重要度を取得完了: {len(importance_df)}個の特徴量")
    return importance_df


def create_importance_bar_chart(importance_df: pd.DataFrame, top_n: int = 10) -> None:
    """特徴量重要度を棒グラフで可視化"""
    print(f"特徴量重要度の棒グラフを作成中（上位{top_n}個）...")

    top_features = importance_df.head(top_n)

    fig = go.Figure(
        data=[
            go.Bar(
                x=top_features["importance"],
                y=top_features["feature"],
                orientation="h",
                marker=dict(
                    color=px.colors.qualitative.Set3[: len(top_features)],
                    line=dict(color="black", width=1),
                ),
                text=[f"{imp:.3f}" for imp in top_features["importance"]],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"特徴量重要度（上位{top_n}個）",
        xaxis_title="重要度",
        yaxis_title="特徴量",
        width=800,
        height=max(400, len(top_features) * 40),
        yaxis=dict(autorange="reversed"),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = OUTPUT_DIR / OUTPUT_FILES["importance_chart"]
    fig.write_html(output_file)
    print(f"特徴量重要度チャートを {output_file} に保存しました")


def analyze_feature_impact(importance_df: pd.DataFrame) -> Dict[str, Any]:
    """特徴量の影響を分析"""
    print("特徴量の影響を分析中...")

    analysis = {
        "most_important_feature": {
            "name": importance_df.iloc[0]["feature"],
            "importance": importance_df.iloc[0]["importance"],
        },
        "top_3_features": [
            {"name": row["feature"], "importance": row["importance"]}
            for _, row in importance_df.head(3).iterrows()
        ],
        "miss_features": [
            {"name": row["feature"], "importance": row["importance"]}
            for _, row in importance_df[
                importance_df["feature"].str.contains("miss", case=False)
            ].iterrows()
        ],
        "score_features": [
            {"name": row["feature"], "importance": row["importance"]}
            for _, row in importance_df[
                importance_df["feature"].str.contains("score", case=False)
            ].iterrows()
        ],
        "importance_stats": {
            "mean": importance_df["importance"].mean(),
            "std": importance_df["importance"].std(),
            "max": importance_df["importance"].max(),
            "min": importance_df["importance"].min(),
        },
    }

    return analysis


def generate_insights_and_recommendations(analysis: Dict[str, Any]) -> None:
    """
    分析結果から洞察と推奨事項を生成する

    Args:
        analysis: 特徴量分析結果
    """
    print("\n=== 特徴量重要度分析結果 ===")

    # 最も重要な特徴量
    top_feature = analysis["most_important_feature"]
    print(
        f"最も重要な特徴量: {top_feature['name']} (重要度: {top_feature['importance']:.3f})"
    )

    # 上位3つの特徴量
    print("\n上位3つの特徴量:")
    for i, feature in enumerate(analysis["top_3_features"], 1):
        print(f"{i}. {feature['name']}: {feature['importance']:.3f}")

    # ミスタイプ関連の特徴量
    if analysis["miss_features"]:
        print("\nミスタイプ関連の特徴量:")
        for feature in analysis["miss_features"]:
            print(f"- {feature['name']}: {feature['importance']:.3f}")
    else:
        print("\nミスタイプ関連の特徴量は見つかりませんでした")

    # スコア関連の特徴量
    if analysis["score_features"]:
        print("\nスコア関連の特徴量:")
        for feature in analysis["score_features"]:
            print(f"- {feature['name']}: {feature['importance']:.3f}")
    else:
        print("\nスコア関連の特徴量は見つかりませんでした")

    # 洞察と推奨事項
    print("\n=== 洞察と推奨事項 ===")

    # ミスタイプの影響について
    miss_features = analysis["miss_features"]
    if miss_features:
        avg_miss_importance = np.mean([f["importance"] for f in miss_features])
        print(f"ミスタイプ関連の特徴量の平均重要度: {avg_miss_importance:.3f}")

        if avg_miss_importance > analysis["importance_stats"]["mean"]:
            print("✅ ミスタイプはスコアに大きな影響を与えています")
            print("💡 推奨: ミスタイプを減らすことでスコア向上が期待できます")
        else:
            print("⚠️ ミスタイプの影響は中程度です")
    else:
        print("❌ ミスタイプ関連の特徴量が見つからないため、影響を評価できません")

    # 過去のスコアの影響について
    score_features = analysis["score_features"]
    if score_features:
        avg_score_importance = np.mean([f["importance"] for f in score_features])
        print(f"スコア関連の特徴量の平均重要度: {avg_score_importance:.3f}")

        if avg_score_importance > analysis["importance_stats"]["mean"]:
            print("✅ 過去のスコアパフォーマンスは重要な予測因子です")
            print("💡 推奨: 一貫したパフォーマンスの維持が重要です")

    # シミュレーションの根拠
    print("\n=== シミュレーションの根拠 ===")
    if miss_features:
        print("ミスタイプを減らせばスコアが伸びる根拠:")
        for feature in miss_features:
            print(f"- {feature['name']}の重要度: {feature['importance']:.3f}")
            if "avg_miss" in feature["name"].lower():
                print("  → 過去のミスタイプ平均が高いほど、スコアが下がる傾向")
            elif "total_miss" in feature["name"].lower():
                print("  → 現在のミスタイプ数が高いほど、スコアが下がる傾向")
    else:
        print(
            "ミスタイプ関連の特徴量が見つからないため、シミュレーションの根拠を提供できません"
        )


def save_analysis_results(
    importance_df: pd.DataFrame, analysis: Dict[str, Any]
) -> None:
    """
    分析結果をCSVファイルに保存する

    Args:
        importance_df: 特徴量重要度のデータフレーム
        analysis: 分析結果
    """
    print("分析結果を保存中...")

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 特徴量重要度をCSVで保存
    csv_file = OUTPUT_DIR / OUTPUT_FILES["importance_csv"]
    importance_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    # 分析結果をテキストファイルで保存
    summary_file = OUTPUT_DIR / OUTPUT_FILES["analysis_summary"]
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== 特徴量重要度分析サマリー ===\n\n")

        f.write("最も重要な特徴量:\n")
        top_feature = analysis["most_important_feature"]
        f.write(f"- {top_feature['name']}: {top_feature['importance']:.3f}\n\n")

        f.write("上位3つの特徴量:\n")
        for i, feature in enumerate(analysis["top_3_features"], 1):
            f.write(f"{i}. {feature['name']}: {feature['importance']:.3f}\n")

        f.write("\nミスタイプ関連の特徴量:\n")
        if analysis["miss_features"]:
            for feature in analysis["miss_features"]:
                f.write(f"- {feature['name']}: {feature['importance']:.3f}\n")
        else:
            f.write("- 見つかりませんでした\n")

        f.write("\nスコア関連の特徴量:\n")
        if analysis["score_features"]:
            for feature in analysis["score_features"]:
                f.write(f"- {feature['name']}: {feature['importance']:.3f}\n")
        else:
            f.write("- 見つかりませんでした\n")

    print("分析結果を output/ ディレクトリに保存しました")


def analyze_feature_importance(
    model: xgb.XGBRegressor, feature_names: List[str]
) -> None:
    """
    特徴量重要度分析のメイン関数

    Args:
        model: 学習済みXGBoostモデル
        feature_names: 特徴量名のリスト
    """
    # 1. 特徴量重要度の取得
    importance_df = get_feature_importance(model, feature_names)

    # 2. 重要度の可視化
    create_importance_bar_chart(importance_df, top_n=10)

    # 3. 特徴量の影響分析
    analysis = analyze_feature_impact(importance_df)

    # 4. 洞察と推奨事項の生成
    generate_insights_and_recommendations(analysis)

    # 5. 結果の保存
    save_analysis_results(importance_df, analysis)

    print("\n特徴量重要度分析が完了しました")


if __name__ == "__main__":
    # テスト用のダミーデータを作成
    import numpy as np

    np.random.seed(42)

    # ダミーの特徴量名
    feature_names = [
        "user_id",
        "score",
        "total_miss",
        "prev_score",
        "avg_score_3",
        "avg_miss_3",
        "score_std_3",
        "max_score_3",
        "min_score_3",
    ]

    # ダミーの重要度（合計が1になるように正規化）
    importance_scores = np.random.random(len(feature_names))
    importance_scores = importance_scores / importance_scores.sum()

    # ダミーモデルの作成
    class DummyModel:
        def __init__(self, feature_importances):
            self.feature_importances_ = feature_importances

    dummy_model = DummyModel(importance_scores)

    # 特徴量重要度分析のテスト
    analyze_feature_importance(dummy_model, feature_names)
