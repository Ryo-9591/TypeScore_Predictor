import logging
from datetime import datetime
import pytz
from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd

# 共通インポート
COMMON_IMPORTS = {
    "standard": ["sys", "Path", "datetime", "logging", "typing", "pytz"],
    "data": ["pandas as pd", "numpy as np"],
    "web": ["dash", "html", "dcc", "Input", "Output", "callback"],
    "plotting": ["plotly.graph_objects as go", "plotly.express as px"],
    "ml": [
        "xgboost as xgb",
        "sklearn.model_selection",
        "sklearn.metrics",
        "sklearn.preprocessing",
    ],
}


# 共通ログ設定
def get_logger(name: str) -> logging.Logger:
    """指定された名前のロガーを取得"""
    return logging.getLogger(name)


def get_jst_time() -> datetime:
    """日本時間の現在時刻を取得"""
    return datetime.now(pytz.timezone("Asia/Tokyo"))


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """日時をフォーマット"""
    return dt.strftime(format_str)


# 共通エラーハンドリング
class AppError(Exception):
    """アプリケーション共通エラー"""

    def __init__(self, message: str, error_type: str = "unknown"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """共通エラーハンドリング"""
    logger = get_logger(__name__)
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg)

    return {
        "status": "error",
        "error": error_msg,
        "timestamp": get_jst_time().isoformat(),
    }


# 共通データ変換
def convert_numpy_types(obj: Any) -> Any:
    """numpy型をPython型に変換"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, "dtype"):
        return str(obj)
    else:
        return obj


# 共通データ処理
def safe_dataframe_operation(
    df: pd.DataFrame, operation: str, **kwargs
) -> pd.DataFrame:
    """安全なデータフレーム操作"""
    try:
        if operation == "fillna":
            return df.fillna(**kwargs)
        elif operation == "dropna":
            return df.dropna(**kwargs)
        elif operation == "sort_values":
            return df.sort_values(**kwargs)
        elif operation == "groupby":
            return df.groupby(**kwargs)
        else:
            raise ValueError(f"不明な操作: {operation}")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"データフレーム操作エラー ({operation}): {e}")
        return df


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """データフレームの検証"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger = get_logger(__name__)
        logger.error(f"必須カラムが見つかりません: {missing_columns}")
        return False
    return True


# 共通設定アクセス
def get_config_value(key: str, default: Any = None) -> Any:
    """設定値を安全に取得"""
    from app.config import globals

    config = globals()
    return config.get(key, default)


# 共通メトリクス計算
def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """共通メトリクス計算"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


# 共通スタイル設定
def get_common_styles() -> Dict[str, Dict[str, Any]]:
    """共通スタイル設定を取得"""
    return {
        "dark_theme": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#ffffff", "size": 12},
        },
        "colors": {
            "primary": "#007bff",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "info": "#17a2b8",
        },
    }
