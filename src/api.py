"""
TypeScore Predictor - FastAPI REST API
予測エンドポイントとデータ管理API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation import prepare_data
from src.feature_engineering import engineer_features
from src.model_training import train_and_evaluate_model

# FastAPIアプリの初期化
app = FastAPI(
    title="TypeScore Predictor API",
    description="タイピングスコア予測のためのREST API",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数（モデルキャッシュ用）
cached_model = None
cached_data = None
cached_metrics = None
cached_analysis_data = None


def load_model():
    """モデルとデータを読み込んでキャッシュ"""
    global cached_model, cached_data, cached_metrics

    if cached_model is None:
        print("モデルを読み込み中...")
        df_final = prepare_data()
        X, y = engineer_features(df_final)
        model, metrics = train_and_evaluate_model(X, y)

        cached_model = model
        cached_data = {
            "df_final": df_final,
            "X": X,
            "y": y,
            "feature_names": list(X.columns),
        }
        cached_metrics = metrics
        print("モデル読み込み完了")
        print(f"キャッシュされたデータのカラム: {list(df_final.columns)}")

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
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "users": "/users",
            "user_stats": "/users/{user_id}/stats",
            "user_timeseries": "/users/{user_id}/timeseries",
            "metrics": "/metrics",
            "retrain": "/retrain",
            "analyze": "/analyze",
            "docs": "/docs",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_score(request: PredictionRequest):
    """スコア予測エンドポイント"""
    try:
        model, data, metrics = load_model()

        # 特徴量を準備
        feature_values = []
        feature_names = data["feature_names"]

        for feature_name in feature_names:
            if feature_name == "user_id":
                # user_idは文字列なので、ハッシュ値に変換して数値化
                import hashlib

                user_id_hash = int(
                    hashlib.md5(request.user_id.encode()).hexdigest()[:8], 16
                )
                feature_values.append(user_id_hash)
            elif feature_name == "prev_score":
                feature_values.append(request.prev_score or 0.0)
            elif feature_name == "avg_score_3":
                feature_values.append(request.avg_score_3 or 0.0)
            elif feature_name == "max_score_3":
                feature_values.append(request.max_score_3 or 0.0)
            elif feature_name == "min_score_3":
                feature_values.append(request.min_score_3 or 0.0)
            elif feature_name == "typing_count":
                feature_values.append(request.typing_count or 0)
            elif feature_name == "avg_miss_3":
                feature_values.append(request.avg_miss_3 or 0.0)
            else:
                feature_values.append(0.0)

        # 予測実行
        X_pred = np.array([feature_values]).reshape(1, -1)
        predicted_score = model.predict(X_pred)[0]

        # 信頼度計算（簡易版：予測値の標準偏差ベース）
        confidence = max(0.0, min(1.0, 1.0 - (metrics["test_rmse"] / 2000.0)))

        # 特徴量重要度
        feature_importance = dict(zip(feature_names, model.feature_importances_))

        return PredictionResponse(
            predicted_score=float(predicted_score),
            confidence=confidence,
            feature_importance=feature_importance,
            model_info={
                "rmse": metrics["test_rmse"],
                "mae": metrics["test_mae"],
                "sample_count": len(data["X"]),
                "feature_count": len(feature_names),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")


@app.get("/users", response_model=List[str])
async def get_users():
    """ユーザー一覧取得"""
    try:
        # キャッシュされたデータからユーザー一覧を取得
        _, cached_data, _ = load_model()
        df_final = cached_data["df_final"]
        users = sorted(
            [str(user_id) for user_id in df_final["user_id"].unique().tolist()]
        )
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ユーザー取得エラー: {str(e)}")


@app.get("/users/{user_id}/timeseries", response_model=UserTimeSeriesResponse)
async def get_user_timeseries(user_id: str):
    """ユーザーの時系列データ取得"""
    try:
        print(f"ユーザー時系列データ取得開始: {user_id}")

        # キャッシュされたデータを使用
        _, cached_data, _ = load_model()
        df_final = cached_data["df_final"]

        user_data = df_final[df_final["user_id"].astype(str) == user_id]

        if len(user_data) == 0:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")

        # 時系列データをソート
        user_data_sorted = user_data.sort_values("created_at")

        return UserTimeSeriesResponse(
            user_id=user_id,
            timestamps=[str(ts) for ts in user_data_sorted["created_at"]],
            scores=[float(score) for score in user_data_sorted["score"]],
            total_misses=[float(miss) for miss in user_data_sorted["total_miss"]],
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"時系列データ取得エラー詳細: {str(e)}")
        raise HTTPException(status_code=500, detail=f"時系列データ取得エラー: {str(e)}")


@app.get("/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """ユーザー統計情報取得"""
    try:
        print(f"ユーザー統計取得開始: {user_id}")

        # キャッシュされたデータを使用
        _, cached_data, _ = load_model()
        df_final = cached_data["df_final"]
        print(f"データ読み込み完了: {df_final.shape}")
        print(f"カラム一覧: {list(df_final.columns)}")

        user_data = df_final[df_final["user_id"].astype(str) == user_id]
        print(f"ユーザーデータ: {len(user_data)}行")

        if len(user_data) == 0:
            print(f"ユーザーが見つかりません: {user_id}")
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")

        # 統計計算
        total_sessions = len(user_data)
        avg_score = user_data["score"].mean()
        max_score = user_data["score"].max()
        min_score = user_data["score"].min()

        # created_atカラムの存在確認
        if "created_at" not in user_data.columns:
            print("created_atカラムが見つかりません")
            raise HTTPException(
                status_code=500, detail="created_atカラムが見つかりません"
            )

        latest_score = user_data.sort_values("created_at")["score"].iloc[-1]

        # 改善傾向の計算
        recent_scores = user_data.sort_values("created_at")["score"].tail(5)
        if len(recent_scores) >= 3:
            trend = (
                "improving"
                if recent_scores.iloc[-1] > recent_scores.iloc[0]
                else "declining"
            )
        else:
            trend = "stable"

        print(f"統計計算完了: sessions={total_sessions}, avg={avg_score}")

        return UserStatsResponse(
            user_id=user_id,
            total_sessions=total_sessions,
            avg_score=float(avg_score),
            max_score=float(max_score),
            min_score=float(min_score),
            latest_score=float(latest_score),
            improvement_trend=trend,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"統計取得エラー詳細: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"統計取得エラー: {str(e)}")


@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """モデル性能指標取得"""
    try:
        _, data, metrics = load_model()

        return ModelMetricsResponse(
            rmse=metrics["test_rmse"],
            mae=metrics["test_mae"],
            sample_count=len(data["X"]),
            feature_count=len(data["feature_names"]),
            last_updated=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"メトリクス取得エラー: {str(e)}")


def run_analysis_process():
    """分析処理を実行して結果を返す"""
    global cached_analysis_data

    if cached_analysis_data is None:
        start_time = datetime.now()

        # データ準備とクリーニング
        df_final = prepare_data()

        # 特徴量エンジニアリング
        X, y = engineer_features(df_final)

        # モデル学習と評価
        model, metrics = train_and_evaluate_model(X, y)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 特徴量重要度
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        # データ情報
        data_info = {
            "total_samples": len(df_final),
            "unique_users": df_final["user_id"].nunique(),
            "feature_count": len(X.columns),
            "score_range": {
                "min": float(df_final["score"].min()),
                "max": float(df_final["score"].max()),
                "mean": float(df_final["score"].mean()),
                "median": float(df_final["score"].median()),
            },
            "training_samples": len(X),
            "test_samples": int(len(X) * 0.2),  # 推定値
        }

        # メトリクス情報
        metrics_info = {
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "target_mae": 200.0,
            "mae_diff": metrics["test_mae"] - 200.0,
            "achievement_status": "未達成" if metrics["test_mae"] > 200.0 else "達成",
        }

        cached_analysis_data = {
            "status": "completed",
            "execution_time": execution_time,
            "metrics": metrics_info,
            "data_info": data_info,
            "feature_importance": feature_importance,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"分析完了 - 実行時間: {execution_time:.2f}秒")

    return cached_analysis_data


@app.post("/analyze", response_model=AnalysisResponse)
async def run_analysis():
    """分析処理実行"""
    try:
        analysis_result = run_analysis_process()
        return AnalysisResponse(**analysis_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析エラー: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """モデル再学習"""
    try:
        global cached_model, cached_data, cached_metrics

        # キャッシュをクリア
        cached_model = None
        cached_data = None
        cached_metrics = None

        # 再学習実行
        model, data, metrics = load_model()

        return {
            "message": "モデル再学習が完了しました",
            "metrics": {
                "rmse": metrics["test_rmse"],
                "mae": metrics["test_mae"],
                "sample_count": len(data["X"]),
                "feature_count": len(data["feature_names"]),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"再学習エラー: {str(e)}")


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    try:
        _, _, _ = load_model()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    import uvicorn

    # 起動時に分析を実行
    print("API起動中...")
    try:
        run_analysis_process()
        print("初期分析完了")
    except Exception as e:
        print(f"初期分析エラー: {str(e)}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
