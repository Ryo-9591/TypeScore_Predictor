"""
予測サービス
スコア予測に関するビジネスロジックを提供
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from app.core import DataProcessor, FeatureEngineer, ModelTrainer

logger = logging.getLogger(__name__)


class PredictionService:
    """予測サービスのメインクラス"""

    def __init__(self):
        """予測サービスの初期化"""
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self._model = None
        self._feature_names = []
        self._is_trained = False

    def train_model(self) -> Dict[str, Any]:
        """
        モデルを学習する

        Returns:
            学習結果の辞書
        """
        logger.info("予測モデルの学習を開始...")

        try:
            # データの準備
            df = self.data_processor.get_processed_data()

            # 特徴量エンジニアリング
            X, y = self.feature_engineer.create_features(df)

            # モデル学習
            model, metrics = self.model_trainer.train_model(X, y)

            # 学習済みモデルを保存
            self._model = model
            self._feature_names = self.feature_engineer.get_feature_names()
            self._is_trained = True

            # 評価結果
            evaluation = self.model_trainer.evaluate_performance()

            result = {
                "status": "success",
                "metrics": metrics,
                "evaluation": evaluation,
                "feature_count": len(self._feature_names),
                "training_samples": len(X),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"モデル学習完了: MAE={metrics['test_mae']:.2f}")
            return result

        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def predict_score(
        self,
        user_id: str,
        prev_score: Optional[float] = None,
        avg_score_3: Optional[float] = None,
        max_score_3: Optional[float] = None,
        min_score_3: Optional[float] = None,
        typing_count: Optional[int] = None,
        avg_miss_3: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        スコアを予測する

        Args:
            user_id: ユーザーID
            prev_score: 前回スコア
            avg_score_3: 過去3回平均スコア
            max_score_3: 過去3回最大スコア
            min_score_3: 過去3回最小スコア
            typing_count: タイピング数
            avg_miss_3: 過去3回平均ミス数

        Returns:
            予測結果の辞書
        """
        if not self._is_trained:
            return {
                "status": "error",
                "error": "モデルが学習されていません",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # 特徴量の準備
            feature_values = self._prepare_features(
                user_id,
                prev_score,
                avg_score_3,
                max_score_3,
                min_score_3,
                typing_count,
                avg_miss_3,
            )

            # 予測実行
            predicted_score = self._model.predict(feature_values.reshape(1, -1))[0]

            # 信頼度計算（簡易版）
            confidence = self._calculate_confidence()

            # 特徴量重要度
            feature_importance = self.model_trainer.get_feature_importance()

            return {
                "status": "success",
                "predicted_score": float(predicted_score),
                "confidence": confidence,
                "feature_importance": feature_importance,
                "model_info": {
                    "rmse": self.model_trainer.metrics.get("test_rmse", 0),
                    "mae": self.model_trainer.metrics.get("test_mae", 0),
                    "feature_count": len(self._feature_names),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _prepare_features(
        self,
        user_id: str,
        prev_score: Optional[float],
        avg_score_3: Optional[float],
        max_score_3: Optional[float],
        min_score_3: Optional[float],
        typing_count: Optional[int],
        avg_miss_3: Optional[float],
    ) -> np.ndarray:
        """予測用の特徴量を準備"""
        feature_values = []

        for feature_name in self._feature_names:
            if feature_name == "user_id_numeric":
                # ユーザーIDの数値化
                user_id_hash = hash(str(user_id)) % 10000
                feature_values.append(user_id_hash)
            elif feature_name == "prev_score":
                feature_values.append(prev_score or 0.0)
            elif feature_name == "avg_score_3":
                feature_values.append(avg_score_3 or 0.0)
            elif feature_name == "max_score_3":
                feature_values.append(max_score_3 or 0.0)
            elif feature_name == "min_score_3":
                feature_values.append(min_score_3 or 0.0)
            elif feature_name == "typing_count":
                feature_values.append(typing_count or 0)
            elif feature_name == "avg_miss_3":
                feature_values.append(avg_miss_3 or 0.0)
            else:
                # その他の特徴量はデフォルト値
                feature_values.append(0.0)

        return np.array(feature_values)

    def _calculate_confidence(self) -> float:
        """予測の信頼度を計算"""
        if not self.model_trainer.metrics:
            return 0.5

        # RMSEベースの信頼度計算
        rmse = self.model_trainer.metrics.get("test_rmse", 1000)
        confidence = max(0.0, min(1.0, 1.0 - (rmse / 2000.0)))
        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self._is_trained:
            return {
                "is_trained": False,
                "message": "モデルが学習されていません",
            }

        result = {
            "is_trained": True,
            "feature_count": len(self._feature_names),
            "sample_count": getattr(self.model_trainer, "training_samples", 0),
            "feature_names": self._feature_names,
            "metrics": self.model_trainer.metrics,
            "evaluation": self.model_trainer.evaluate_performance(),
        }

        print(f"DEBUG: get_model_info result = {result}")
        return result

    def retrain_model(self) -> Dict[str, Any]:
        """
        モデルを再学習する

        Returns:
            再学習結果の辞書
        """
        logger.info("モデル再学習を開始...")

        # キャッシュをクリア
        self.data_processor._cached_data = None

        # 再学習実行
        result = self.train_model()

        logger.info("モデル再学習完了")
        return result
