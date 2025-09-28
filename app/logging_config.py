"""
ログ設定の初期化モジュール
アプリケーション全体のログ設定を管理
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime
import pytz

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import LOG_CONFIG, PREDICTION_REPORT_CONFIG, PROJECT_ROOT


class JSTFormatter(logging.Formatter):
    """日本時間（JST）でログをフォーマットするカスタムフォーマッター"""

    def formatTime(self, record, datefmt=None):
        # 日本時間に変換
        jst = pytz.timezone("Asia/Tokyo")
        dt = datetime.fromtimestamp(record.created, tz=jst)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_logging():
    """ログ設定を初期化"""

    # ログディレクトリの作成
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 現在の日付を取得（日本時間）
    jst = pytz.timezone("Asia/Tokyo")
    today = datetime.now(jst).strftime("%Y-%m-%d")

    # 日付付きログファイル名を生成
    app_log_file = PROJECT_ROOT / "logs" / f"app_{today}.log"
    prediction_log_file = PROJECT_ROOT / "logs" / f"prediction_report_{today}.log"

    # 基本ログ設定
    # 日次ローテーションファイルハンドラー
    file_handler = logging.handlers.TimedRotatingFileHandler(
        app_log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # 30日分保持
        encoding="utf-8",
    )
    file_handler.setFormatter(JSTFormatter(LOG_CONFIG["format"]))

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSTFormatter(LOG_CONFIG["format"]))

    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_CONFIG["level"]))
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 予測精度レポート用ロガーの設定（日付付きファイル）
    if PREDICTION_REPORT_CONFIG["enabled"]:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file_pattern = str(PREDICTION_REPORT_CONFIG["file_pattern"])
        prediction_log_file = Path(log_file_pattern.format(date=today))

        report_logger = logging.getLogger("prediction_report")
        report_logger.setLevel(getattr(logging, PREDICTION_REPORT_CONFIG["level"]))

        # 日次ローテーションファイルハンドラーの設定
        report_handler = logging.handlers.TimedRotatingFileHandler(
            prediction_log_file,
            when="midnight",
            interval=1,
            backupCount=30,  # 30日分保持
            encoding="utf-8",
        )
        report_handler.setFormatter(JSTFormatter(PREDICTION_REPORT_CONFIG["format"]))

        # 既存のハンドラーをクリアして新しいハンドラーを追加
        report_logger.handlers.clear()
        report_logger.addHandler(report_handler)
        report_logger.propagate = False  # 親ロガーに伝播しない

    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_CONFIG["level"]))

    print("ログ設定が完了しました。")
    print(f"アプリケーションログ: {app_log_file}")
    print(f"予測精度レポートログ: {prediction_log_file}")


def get_logger(name: str) -> logging.Logger:
    """指定された名前のロガーを取得"""
    return logging.getLogger(name)


def get_report_logger() -> logging.Logger:
    """予測精度レポート用のロガーを取得"""
    return logging.getLogger("prediction_report")


if __name__ == "__main__":
    setup_logging()

    # テストログ出力
    logger = get_logger(__name__)
    report_logger = get_report_logger()

    logger.info("アプリケーションログのテスト")
    report_logger.info("予測精度レポートログのテスト")
