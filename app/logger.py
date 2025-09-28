"""
ログ機能を提供するモジュール
"""

import logging
import os
from datetime import datetime
from typing import Optional
from config import config


class Logger:
    """ログ管理クラス"""

    def __init__(self, name: str = "TypeScorePredictor"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.app.log_level))

        # 既存のハンドラーをクリア
        self.logger.handlers.clear()

        # フォーマッターの設定
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # ファイルハンドラー（ログディレクトリが存在する場合）
        log_dir = "logs"
        if os.path.exists(log_dir):
            log_file = os.path.join(
                log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """情報ログ"""
        self.logger.info(message)

    def warning(self, message: str):
        """警告ログ"""
        self.logger.warning(message)

    def error(self, message: str):
        """エラーログ"""
        self.logger.error(message)

    def debug(self, message: str):
        """デバッグログ"""
        self.logger.debug(message)

    def exception(self, message: str):
        """例外ログ（スタックトレース付き）"""
        self.logger.exception(message)


# グローバルロガーインスタンス
logger = Logger()
