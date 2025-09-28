"""
ログ出力用のユーティリティ関数
numpy型をPython型に変換し、読みやすいテキスト形式でログを出力
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from datetime import datetime
import pytz


def convert_numpy_types(obj: Any) -> Any:
    """
    numpy型をPython型に変換する

    Args:
        obj: 変換対象のオブジェクト

    Returns:
        変換されたオブジェクト
    """
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
    elif hasattr(obj, "dtype"):  # pandas dtype
        return str(obj)
    else:
        return obj


def format_log_text(data: Dict[str, Any], event_type: str) -> str:
    """
    読みやすいテキスト形式でログをフォーマット

    Args:
        data: ログデータ
        event_type: イベントタイプ

    Returns:
        フォーマットされたテキスト文字列
    """
    # 日本時間でタイムスタンプを生成
    jst = pytz.timezone('Asia/Tokyo')
    timestamp = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")

    # numpy型をPython型に変換
    converted_data = convert_numpy_types(data)

    lines = [f"[{timestamp}] {event_type}"]
    lines.append("=" * 60)

    # データを再帰的にフォーマット
    _format_dict(converted_data, lines, indent=0)

    lines.append("=" * 60)
    lines.append("")  # 空行で区切り

    return "\n".join(lines)


def _format_dict(data: Any, lines: List[str], indent: int = 0) -> None:
    """
    辞書やリストを再帰的にフォーマット

    Args:
        data: フォーマットするデータ
        lines: 出力行のリスト
        indent: インデントレベル
    """
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                _format_dict(value, lines, indent + 1)
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}[{i}]:")
                _format_dict(item, lines, indent + 1)
            else:
                lines.append(f"{prefix}[{i}]: {item}")
    else:
        lines.append(f"{prefix}{data}")


def safe_text_log(data: Dict[str, Any], event_type: str) -> str:
    """
    安全にテキスト形式でログを生成

    Args:
        data: ログデータ
        event_type: イベントタイプ

    Returns:
        フォーマットされたテキスト文字列
    """
    return format_log_text(data, event_type)
