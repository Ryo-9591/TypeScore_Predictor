import sys
from pathlib import Path
import logging
from datetime import datetime
import pytz
from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Docker環境での追加パス設定
APP_ROOT = Path(__file__).parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

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


# 共通キャッシュ管理
class CacheManager:
    """共通キャッシュ管理クラス"""

    def __init__(self):
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """キャッシュから値を取得"""
        if key not in self._cache:
            return None

        # 期限チェック
        if key in self._timestamps:
            age = (datetime.now() - self._timestamps[key]).total_seconds()
            if age > max_age_seconds:
                self.clear(key)
                return None

        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """キャッシュに値を設定"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    def clear(self, key: Optional[str] = None) -> None:
        """キャッシュをクリア"""
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()

    def clear_expired(self, max_age_seconds: int = 300) -> int:
        """期限切れのキャッシュをクリア"""
        now = datetime.now()
        expired_keys = []

        for key, timestamp in self._timestamps.items():
            if (now - timestamp).total_seconds() > max_age_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self.clear(key)

        return len(expired_keys)


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
    try:
        from app.config import globals

        config = globals()
        return config.get(key, default)
    except ImportError:
        return default


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


# デコレータ
def cached_property(func):
    """キャッシュ付きプロパティデコレータ"""
    attr_name = f"_cached_{func.__name__}"

    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return property(wrapper)


def log_execution_time(func):
    """実行時間をログ出力するデコレータ"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} 実行時間: {execution_time:.2f}秒")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} エラー ({execution_time:.2f}秒): {e}")
            raise

    return wrapper
