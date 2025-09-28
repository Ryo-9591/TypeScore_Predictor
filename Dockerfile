# Python 3.11の公式イメージを使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app/ ./app/
COPY data/ ./data/

# ログディレクトリの作成
RUN mkdir -p /app/logs

# データディレクトリとログディレクトリの権限設定
RUN chmod -R 755 /app/data
RUN chmod -R 755 /app/logs

# デフォルトコマンド
CMD ["python", "app/main.py"]
