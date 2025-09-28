"""
ログ出力用のユーティリティ関数（最適化版）
共通ユーティリティとの統合により重複を削除
"""

try:
    from app.utils.common import convert_numpy_types, get_jst_time, format_datetime
except ImportError:
    # Docker環境でのフォールバック
    import sys
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from utils.common import convert_numpy_types, get_jst_time, format_datetime
from typing import Any, Dict, List


def format_log_text(data: Dict[str, Any], event_type: str) -> str:
    """読みやすいテキスト形式でログをフォーマット"""
    timestamp = format_datetime(get_jst_time())
    converted_data = convert_numpy_types(data)

    lines = [f"[{timestamp}] {event_type}", "=" * 60]
    _format_dict(converted_data, lines, indent=0)
    lines.extend(["=" * 60, ""])  # 区切り線と空行

    return "\n".join(lines)


def _format_dict(data: Any, lines: List[str], indent: int = 0) -> None:
    """辞書やリストを再帰的にフォーマット"""
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
    """安全にテキスト形式でログを生成"""
    return format_log_text(data, event_type)
