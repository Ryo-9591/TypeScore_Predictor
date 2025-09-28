#!/usr/bin/env python3
"""
ログ設定の初期化スクリプト
予測精度レポート用のログディレクトリとファイルを作成
"""

import os
import logging
from pathlib import Path
from app.config import PROJECT_ROOT, PREDICTION_REPORT_CONFIG

# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_logging():
    """ログディレクトリとファイルを設定"""

    # ログディレクトリの作成
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger.info(f"ログディレクトリを作成しました: {logs_dir}")

    # 予測精度レポート用ログファイルの作成
    if PREDICTION_REPORT_CONFIG["enabled"]:
        report_log_file = PREDICTION_REPORT_CONFIG["file"]
        report_log_file.parent.mkdir(parents=True, exist_ok=True)

        # ログファイルが存在しない場合は作成
        if not report_log_file.exists():
            report_log_file.touch()
            logger.info(
                f"予測精度レポートログファイルを作成しました: {report_log_file}"
            )
        else:
            logger.info(
                f"予測精度レポートログファイルは既に存在します: {report_log_file}"
            )

    # 一般的なアプリケーションログファイルの作成
    app_log_file = PROJECT_ROOT / "logs" / "app.log"
    if not app_log_file.exists():
        app_log_file.touch()
        logger.info(f"アプリケーションログファイルを作成しました: {app_log_file}")
    else:
        logger.info(f"アプリケーションログファイルは既に存在します: {app_log_file}")

    logger.info("ログ設定の初期化が完了しました。")


if __name__ == "__main__":
    setup_logging()
