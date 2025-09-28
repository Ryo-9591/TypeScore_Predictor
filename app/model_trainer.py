import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from logger import logger
from config import config


class ModelTrainer:
    """機械学習モデルの訓練を担当するクラス"""

    def __init__(self, merged_data: pd.DataFrame, feature_columns: List[str]):
        """
        Args:
            merged_data (pd.DataFrame): 前処理済みの結合データ
            feature_columns (List[str]): 特徴量の列名リスト
        """
        self.merged_data = merged_data
        self.feature_columns = feature_columns
        self.models: Dict[str, RandomForestRegressor] = {}
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        self.max_workers = min(
            4, len([1, 2, 3]) * len([1, 2])
        )  # 並列処理の最大ワーカー数

    def train_models(self) -> bool:
        """各モード（難易度×言語）ごとにモデルを訓練（並列処理対応）

        Returns:
            bool: 訓練成功の場合True
        """
        logger.info("モデルの訓練を開始します...")

        # 訓練対象のモードを準備
        training_tasks = []
        for diff_id in [1, 2, 3]:
            for lang_id in [1, 2]:
                mode_key = f"diff_{diff_id}_lang_{lang_id}"
                mode_data = self.merged_data[
                    (self.merged_data["diff_id"] == diff_id)
                    & (self.merged_data["lang_id"] == lang_id)
                ].copy()

                if len(mode_data) >= config.model.min_data_threshold:
                    training_tasks.append((mode_key, mode_data))
                else:
                    logger.warning(
                        f"モード {mode_key}: データ不足のためスキップ ({len(mode_data)}件)"
                    )

        if not training_tasks:
            logger.error("訓練可能なモードがありません")
            return False

        # 並列処理でモデルを訓練
        trained_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクを並列実行
            future_to_mode = {
                executor.submit(self._train_single_model, mode_key, mode_data): mode_key
                for mode_key, mode_data in training_tasks
            }

            # 結果を収集
            for future in as_completed(future_to_mode):
                mode_key = future_to_mode[future]
                try:
                    success = future.result()
                    if success:
                        trained_count += 1
                        logger.info(f"モード {mode_key}: 訓練完了")
                    else:
                        logger.error(f"モード {mode_key}: 訓練失敗")
                except Exception as e:
                    logger.exception(
                        f"モード {mode_key}: 訓練中にエラーが発生しました: {e}"
                    )

        logger.info(
            f"モデルの訓練が完了しました: {trained_count}個のモデルが訓練されました"
        )
        return trained_count > 0

    def _train_single_model(self, mode_key: str, mode_data: pd.DataFrame) -> bool:
        """単一のモードに対するモデルを訓練

        Args:
            mode_key (str): モードキー
            mode_data (pd.DataFrame): モードのデータ

        Returns:
            bool: 訓練成功の場合True
        """
        try:
            X = mode_data[self.feature_columns]
            y = mode_data["score"]

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.model.test_size,
                random_state=config.model.random_state,
            )

            # モデルの訓練（設定を使用）
            model = RandomForestRegressor(
                n_estimators=config.model.n_estimators,
                max_depth=config.model.max_depth,
                min_samples_split=config.model.min_samples_split,
                random_state=config.model.random_state,
                n_jobs=-1,  # 並列処理を有効化
            )
            model.fit(X_train, y_train)

            # 予測と評価
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 結果を保存（スレッドセーフな方法）
            self.models[mode_key] = model
            self.model_performance[mode_key] = {
                "mse": float(mse),
                "r2": float(r2),
                "data_size": len(mode_data),
                "test_size": len(X_test),
                "feature_importance": dict(
                    zip(self.feature_columns, model.feature_importances_.astype(float))
                ),
            }

            logger.debug(f"モード {mode_key}: 訓練完了 (R²={r2:.3f}, MSE={mse:.1f})")
            return True

        except Exception as e:
            logger.exception(f"モード {mode_key} の訓練に失敗しました: {e}")
            return False

    def get_user_predictions(
        self, user_id: str, diff_id: int, lang_id: int
    ) -> Optional[Dict[str, Any]]:
        """特定ユーザーのスコア予測

        Args:
            user_id (str): ユーザーID
            diff_id (int): 難易度ID
            lang_id (int): 言語ID

        Returns:
            Optional[Dict[str, Any]]: 予測結果の辞書、失敗時はNone
        """
        mode_key = f"diff_{diff_id}_lang_{lang_id}"

        if mode_key not in self.models:
            logger.warning(f"モデルが見つかりません: {mode_key}")
            return None

        try:
            # ユーザーの最新データを取得
            user_data = self.merged_data[
                (self.merged_data["user_id"] == user_id)
                & (self.merged_data["diff_id"] == diff_id)
                & (self.merged_data["lang_id"] == lang_id)
            ]

            if len(user_data) == 0:
                logger.warning(
                    f"ユーザー {user_id} のモード {mode_key} のデータが見つかりません"
                )
                return None

            # 最新のレコードを使用
            latest_record = user_data.iloc[-1]
            X = latest_record[self.feature_columns].values.reshape(1, -1)
            prediction = self.models[mode_key].predict(X)[0]

            return {
                "predicted_score": float(prediction),
                "actual_score": float(latest_record["score"]),
                "username": str(latest_record["username"]),
                "difficulty": str(latest_record["difficulty_label"]),
                "language": str(latest_record["language_label"]),
                "model_performance": self.model_performance[mode_key],
            }

        except Exception as e:
            logger.exception(f"ユーザー {user_id} の予測でエラーが発生しました: {e}")
            return None

    def get_models(self) -> Dict[str, RandomForestRegressor]:
        """訓練済みモデルを返す

        Returns:
            Dict[str, RandomForestRegressor]: モードキーをキーとするモデル辞書
        """
        return self.models

    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """モデル性能情報を返す

        Returns:
            Dict[str, Dict[str, Any]]: モードキーをキーとする性能情報辞書
        """
        return self.model_performance

    def get_feature_importance(
        self, mode_key: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """特徴量重要度を返す

        Args:
            mode_key (Optional[str]): モードキー、Noneの場合は全てのモード

        Returns:
            Optional[Dict[str, float]]: 特徴量重要度辞書
        """
        if mode_key:
            if mode_key in self.model_performance:
                return self.model_performance[mode_key]["feature_importance"]
            return None
        else:
            return {
                mode: perf["feature_importance"]
                for mode, perf in self.model_performance.items()
            }

    def get_available_modes(self) -> List[str]:
        """利用可能なモードのリストを返す

        Returns:
            List[str]: モードキーのリスト
        """
        return list(self.models.keys())

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルのサマリー情報を返す

        Returns:
            Dict[str, Any]: サマリー情報
        """
        if not self.model_performance:
            return {"total_models": 0}

        r2_scores = [perf["r2"] for perf in self.model_performance.values()]
        mse_scores = [perf["mse"] for perf in self.model_performance.values()]

        return {
            "total_models": len(self.models),
            "avg_r2": float(np.mean(r2_scores)),
            "best_r2": float(np.max(r2_scores)),
            "worst_r2": float(np.min(r2_scores)),
            "avg_mse": float(np.mean(mse_scores)),
            "best_mse": float(np.min(mse_scores)),
            "worst_mse": float(np.max(mse_scores)),
        }
