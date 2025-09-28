import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from logger import logger


class DataVisualizer:
    """データの可視化を担当するクラス"""

    def __init__(self, merged_data: pd.DataFrame, model_performance: Dict[str, Any]):
        """
        Args:
            merged_data (pd.DataFrame): 前処理済みの結合データ
            model_performance (Dict[str, Any]): モデル性能情報
        """
        self.merged_data = merged_data
        self.model_performance = model_performance
        self._figure_cache: Dict[str, go.Figure] = {}  # グラフのキャッシュ

    def create_correlation_plot(self, diff_id: int, lang_id: int) -> go.Figure:
        """ミスタイプとスコアの相関プロットを作成

        Args:
            diff_id (int): 難易度ID
            lang_id (int): 言語ID

        Returns:
            go.Figure: 相関プロット
        """
        cache_key = f"correlation_{diff_id}_{lang_id}"

        # キャッシュをチェック
        if cache_key in self._figure_cache:
            logger.debug(f"相関プロットのキャッシュを使用: {cache_key}")
            return self._figure_cache[cache_key]

        try:
            correlation_data = self.merged_data[
                (self.merged_data["diff_id"] == diff_id)
                & (self.merged_data["lang_id"] == lang_id)
            ]

            if len(correlation_data) == 0:
                logger.warning(
                    f"相関プロット用のデータがありません: 難易度{diff_id}, 言語{lang_id}"
                )
                fig = self._create_empty_plot("データがありません")
                return fig

            fig = px.scatter(
                correlation_data,
                x="total_misses",
                y="score",
                color="username",
                title=f"ミスタイプ数とスコアの相関 (難易度{diff_id} - 言語{lang_id})",
                labels={"total_misses": "総ミスタイプ数", "score": "スコア"},
                hover_data=["accuracy", "typing_count"],
            )

            fig.update_layout(
                height=400,
                showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(color="white"),
                yaxis=dict(color="white"),
            )

            # キャッシュに保存
            self._figure_cache[cache_key] = fig
            logger.debug(f"相関プロットを作成しました: {cache_key}")
            return fig

        except Exception as e:
            logger.exception(f"相関プロットの作成中にエラーが発生しました: {e}")
            return self._create_empty_plot("エラーが発生しました")

    def create_model_performance_plot(self) -> go.Figure:
        """モデル性能プロットを作成

        Returns:
            go.Figure: モデル性能プロット
        """
        cache_key = "model_performance"

        # キャッシュをチェック
        if cache_key in self._figure_cache:
            logger.debug("モデル性能プロットのキャッシュを使用")
            return self._figure_cache[cache_key]

        try:
            if not self.model_performance:
                logger.warning("モデル性能データがありません")
                fig = self._create_empty_plot("モデル性能データがありません")
                return fig

            performance_data = []
            for mode_key, perf in self.model_performance.items():
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

            # R²スコアのバープロット
            fig = px.bar(
                performance_df,
                x="mode",
                y="R²",
                title="モデル性能 (R²スコア)",
                labels={"mode": "モード", "R²": "R²スコア"},
                color="R²",
                color_continuous_scale="Viridis",
            )

            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(color="white"),
                yaxis=dict(color="white"),
            )

            # キャッシュに保存
            self._figure_cache[cache_key] = fig
            logger.debug("モデル性能プロットを作成しました")
            return fig

        except Exception as e:
            logger.exception(f"モデル性能プロットの作成中にエラーが発生しました: {e}")
            return self._create_empty_plot("エラーが発生しました")

    def _create_empty_plot(self, message: str) -> go.Figure:
        """空のプロットを作成

        Args:
            message (str): 表示するメッセージ

        Returns:
            go.Figure: 空のプロット
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="white"),
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        return fig

    def create_user_comparison_plot(self, diff_id, lang_id):
        """ユーザー比較プロットを作成"""
        user_comparison_data = (
            self.merged_data[
                (self.merged_data["diff_id"] == diff_id)
                & (self.merged_data["lang_id"] == lang_id)
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

        if len(user_comparison_data) == 0:
            # データがない場合の空のグラフ
            fig = go.Figure()
            fig.add_annotation(
                text="データがありません",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        user_comparison_data.columns = [
            "avg_score",
            "std_score",
            "count",
            "avg_accuracy",
            "avg_misses",
        ]
        user_comparison_data = user_comparison_data.reset_index()

        fig = px.bar(
            user_comparison_data,
            x="username",
            y="avg_score",
            error_y="std_score",
            title=f"ユーザー別平均スコア比較 (難易度{diff_id} - 言語{lang_id})",
            labels={"username": "ユーザー名", "avg_score": "平均スコア"},
            color="avg_score",
            color_continuous_scale="Blues",
        )

        fig.update_layout(height=400, xaxis_tickangle=-45)

        return fig

    def create_feature_importance_plot(self, mode_key):
        """特徴量重要度プロットを作成"""
        if mode_key not in self.model_performance:
            # データがない場合の空のグラフ
            fig = go.Figure()
            fig.add_annotation(
                text="特徴量重要度データがありません",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        importance = self.model_performance[mode_key]["feature_importance"]

        # 重要度を降順でソート
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)

        fig = px.bar(
            x=importances,
            y=features,
            orientation="h",
            title=f"特徴量重要度 ({mode_key})",
            labels={"x": "重要度", "y": "特徴量"},
            color=importances,
            color_continuous_scale="Reds",
        )

        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})

        return fig

    def create_score_distribution_plot(self, diff_id, lang_id):
        """スコア分布プロットを作成"""
        score_data = self.merged_data[
            (self.merged_data["diff_id"] == diff_id)
            & (self.merged_data["lang_id"] == lang_id)
        ]

        if len(score_data) == 0:
            # データがない場合の空のグラフ
            fig = go.Figure()
            fig.add_annotation(
                text="データがありません",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = px.histogram(
            score_data,
            x="score",
            nbins=30,
            title=f"スコア分布 (難易度{diff_id} - 言語{lang_id})",
            labels={"score": "スコア", "count": "頻度"},
            color_discrete_sequence=["skyblue"],
        )

        fig.update_layout(height=400)

        return fig
