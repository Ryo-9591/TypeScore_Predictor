"""
TypeScore Predictor - FastAPI REST API
新しいアーキテクチャに基づくAPIサーバー
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

# プロジェクトルートをPythonパスに追加
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 新しいアーキテクチャのインポート
from app.services import PredictionService, UserService, AnalysisService
from app.config import API_CONFIG, SECURITY_CONFIG
from app.logging_config import get_logger, setup_logging

# ログ設定の初期化
setup_logging()

# ロガーの設定
logger = get_logger(__name__)

# FastAPIアプリの初期化
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=SECURITY_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=SECURITY_CONFIG["cors_methods"],
    allow_headers=SECURITY_CONFIG["cors_headers"],
)

# サービス層の初期化
prediction_service = PredictionService()
user_service = UserService()
analysis_service = AnalysisService()

# グローバル変数（モデルキャッシュ用）
cached_model = None
cached_data = None
cached_metrics = None
cached_analysis_data = None


def load_model():
    """モデルとデータを読み込んでキャッシュ"""
    global cached_model, cached_data, cached_metrics

    if cached_model is None:
        logger.info("モデルを読み込み中...")
        try:
            result = prediction_service.train_model()
            if result["status"] == "success":
                cached_model = prediction_service._model
                cached_data = {
                    "feature_names": prediction_service._feature_names,
                    "is_trained": prediction_service._is_trained,
                }
                cached_metrics = result["metrics"]
                logger.info("モデル読み込み完了")
            else:
                raise Exception(result.get("error", "モデル学習に失敗しました"))
        except Exception as e:
            logger.error(f"モデル学習エラー: {e}")
            raise

    return cached_model, cached_data, cached_metrics


# Pydanticモデル定義
class PredictionRequest(BaseModel):
    user_id: str
    prev_score: Optional[float] = None
    avg_score_3: Optional[float] = None
    max_score_3: Optional[float] = None
    min_score_3: Optional[float] = None
    typing_count: Optional[int] = None
    avg_miss_3: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_score: float
    confidence: float
    feature_importance: Dict[str, float]
    model_info: Dict[str, Any]


class UserStatsResponse(BaseModel):
    user_id: str
    total_sessions: int
    avg_score: float
    max_score: float
    min_score: float
    latest_score: float
    improvement_trend: str


class UserTimeSeriesResponse(BaseModel):
    user_id: str
    timestamps: List[str]
    scores: List[float]
    total_misses: List[float]


class ModelMetricsResponse(BaseModel):
    rmse: float
    mae: float
    sample_count: int
    feature_count: int
    last_updated: str


class AnalysisResponse(BaseModel):
    status: str
    execution_time: float
    metrics: Dict[str, Any]
    data_info: Dict[str, Any]
    feature_importance: Dict[str, float]
    timestamp: str


# APIエンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "TypeScore Predictor API",
        "version": API_CONFIG["version"],
        "endpoints": {
            "predict": "/predict",
            "users": "/users",
            "user_stats": "/users/{user_id}/stats",
            "user_timeseries": "/users/{user_id}/timeseries",
            "metrics": "/metrics",
            "retrain": "/retrain",
            "analyze": "/analyze",
            "reports_daily": "/reports/daily",
            "reports_comprehensive": "/reports/comprehensive",
            "docs": "/docs",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_score(request: PredictionRequest):
    """スコア予測エンドポイント"""
    try:
        # 予測サービスを使用してスコアを予測
        result = prediction_service.predict_score(
            user_id=request.user_id,
            prev_score=request.prev_score,
            avg_score_3=request.avg_score_3,
            max_score_3=request.max_score_3,
            min_score_3=request.min_score_3,
            typing_count=request.typing_count,
            avg_miss_3=request.avg_miss_3,
        )

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error"])

        return PredictionResponse(
            predicted_score=result["predicted_score"],
            confidence=result["confidence"],
            feature_importance=result["feature_importance"],
            model_info=result["model_info"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")


@app.get("/users", response_model=List[str])
async def get_users():
    """ユーザー一覧取得"""
    try:
        users = user_service.get_all_users()
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ユーザー取得エラー: {str(e)}")


@app.get("/users/{user_id}/timeseries", response_model=UserTimeSeriesResponse)
async def get_user_timeseries(user_id: str):
    """ユーザーの時系列データ取得"""
    try:
        timeseries_data = user_service.get_user_timeseries(user_id)

        if not timeseries_data:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")

        return UserTimeSeriesResponse(
            user_id=timeseries_data["user_id"],
            timestamps=timeseries_data["timestamps"],
            scores=timeseries_data["scores"],
            total_misses=timeseries_data["total_misses"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"時系列データ取得エラー: {str(e)}")


@app.get("/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """ユーザー統計情報取得"""
    try:
        user_stats = user_service.get_user_stats(user_id)

        if not user_stats:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")

        return UserStatsResponse(
            user_id=user_stats["user_id"],
            total_sessions=user_stats["total_sessions"],
            avg_score=user_stats["avg_score"],
            max_score=user_stats["max_score"],
            min_score=user_stats["min_score"],
            latest_score=user_stats["latest_score"],
            improvement_trend=user_stats["trend"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計取得エラー: {str(e)}")


@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """モデル性能指標取得"""
    try:
        model_info = prediction_service.get_model_info()
        logger.debug(f"モデル情報: {model_info}")

        if not model_info["is_trained"]:
            raise HTTPException(status_code=500, detail="モデルが学習されていません")

        metrics = model_info["metrics"]
        logger.debug(f"メトリクス: {metrics}")

        return ModelMetricsResponse(
            rmse=metrics["test_rmse"],
            mae=metrics["test_mae"],
            sample_count=model_info["sample_count"],
            feature_count=model_info["feature_count"],
            last_updated=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"メトリクス取得エラー: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def run_analysis():
    """分析処理実行"""
    try:
        analysis_result = analysis_service.run_full_analysis()

        if analysis_result["status"] != "completed":
            raise HTTPException(
                status_code=500,
                detail=analysis_result.get("error", "分析に失敗しました"),
            )

        return AnalysisResponse(
            status=analysis_result["status"],
            execution_time=analysis_result["execution_time"],
            metrics=analysis_result["metrics"],
            data_info=analysis_result["data_info"],
            feature_importance=analysis_result["feature_importance"],
            timestamp=analysis_result["timestamp"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析エラー: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """モデル再学習"""
    try:
        result = prediction_service.retrain_model()

        if result["status"] != "success":
            raise HTTPException(
                status_code=500, detail=result.get("error", "再学習に失敗しました")
            )

        return {
            "message": "モデル再学習が完了しました",
            "metrics": result["metrics"],
            "timestamp": result["timestamp"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"再学習エラー: {str(e)}")


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    try:
        model_info = prediction_service.get_model_info()
        return {
            "status": "healthy" if model_info["is_trained"] else "unhealthy",
            "model_trained": model_info["is_trained"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/reports/daily")
async def get_daily_report():
    """日次予測精度レポート取得"""
    try:
        result = prediction_service.generate_daily_report()

        if result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "レポート生成に失敗しました"),
            )

        return result["report"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"日次レポート取得エラー: {str(e)}")


@app.get("/reports/comprehensive")
async def get_comprehensive_report():
    """包括的予測精度レポート取得"""
    try:
        result = analysis_service.generate_comprehensive_report()

        if result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "レポート生成に失敗しました"),
            )

        return result["report"]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"包括的レポート取得エラー: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # 起動時に分析を実行
    logger.info("API起動中...")
    try:
        analysis_result = analysis_service.run_full_analysis()
        if analysis_result["status"] == "completed":
            logger.info("初期分析完了")
        else:
            logger.error(
                f"初期分析エラー: {analysis_result.get('error', '不明なエラー')}"
            )
    except Exception as e:
        logger.error(f"初期分析エラー: {str(e)}")

    uvicorn.run(app, host=API_CONFIG["host"], port=API_CONFIG["port"])
