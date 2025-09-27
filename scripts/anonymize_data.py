#!/usr/bin/env python3
"""
データ匿名化スクリプト
個人情報を含むCSVファイルから個人名とメールアドレスを匿名化する
"""

import pandas as pd
import hashlib
import random
import string
from pathlib import Path
import sys

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DATA_DIR


def generate_anonymous_username(original_name, user_id):
    """
    元の名前から匿名化されたユーザー名を生成する

    Args:
        original_name: 元の名前
        user_id: ユーザーID

    Returns:
        str: 匿名化されたユーザー名
    """
    # ユーザーIDの一部を使用して一意なハッシュを生成
    hash_obj = hashlib.md5(user_id.encode())
    hash_hex = hash_obj.hexdigest()[:8]

    # ランダムな文字列を生成
    random_suffix = "".join(random.choices(string.ascii_lowercase, k=4))

    return f"user_{hash_hex}_{random_suffix}"


def generate_anonymous_email(username):
    """
    匿名化されたメールアドレスを生成する

    Args:
        username: 匿名化されたユーザー名

    Returns:
        str: 匿名化されたメールアドレス
    """
    return f"{username}@example.com"


def anonymize_user_data():
    """
    ユーザーデータを匿名化する
    """
    print("ユーザーデータの匿名化を開始します...")

    # 元のファイルをバックアップ
    user_file = DATA_DIR / "m_user.csv"
    backup_file = DATA_DIR / "m_user_backup.csv"

    if not user_file.exists():
        print(f"エラー: {user_file} が見つかりません")
        return

    # バックアップを作成
    df_original = pd.read_csv(user_file)
    df_original.to_csv(backup_file, index=False)
    print(f"バックアップファイルを作成しました: {backup_file}")

    # データを読み込み
    df = pd.read_csv(user_file)

    # 匿名化処理
    df_anonymized = df.copy()

    # username列を匿名化
    if "username" in df_anonymized.columns:
        df_anonymized["username"] = df_anonymized.apply(
            lambda row: generate_anonymous_username(row["username"], row["user_id"]),
            axis=1,
        )
        print("username列を匿名化しました")

    # email列を匿名化
    if "email" in df_anonymized.columns:
        df_anonymized["email"] = df_anonymized["username"].apply(
            generate_anonymous_email
        )
        print("email列を匿名化しました")

    # 匿名化されたデータを保存
    df_anonymized.to_csv(user_file, index=False)
    print(f"匿名化されたデータを保存しました: {user_file}")

    # 匿名化の結果を表示
    print("\n匿名化の結果:")
    print("元のデータ（最初の3行）:")
    print(df[["user_id", "username", "email"]].head(3))
    print("\n匿名化後のデータ（最初の3行）:")
    print(df_anonymized[["user_id", "username", "email"]].head(3))

    print(f"\n匿名化が完了しました。")
    print(f"元のデータは {backup_file} にバックアップされています。")


def restore_original_data():
    """
    バックアップから元のデータを復元する
    """
    print("元のデータの復元を開始します...")

    user_file = DATA_DIR / "m_user.csv"
    backup_file = DATA_DIR / "m_user_backup.csv"

    if not backup_file.exists():
        print(f"エラー: バックアップファイル {backup_file} が見つかりません")
        return

    # バックアップから復元
    df_backup = pd.read_csv(backup_file)
    df_backup.to_csv(user_file, index=False)

    print(f"元のデータを復元しました: {user_file}")


def main():
    """
    メイン関数
    """
    import argparse

    parser = argparse.ArgumentParser(description="データ匿名化ツール")
    parser.add_argument(
        "action",
        choices=["anonymize", "restore"],
        help="実行するアクション: anonymize (匿名化) または restore (復元)",
    )

    args = parser.parse_args()

    if args.action == "anonymize":
        anonymize_user_data()
    elif args.action == "restore":
        restore_original_data()


if __name__ == "__main__":
    main()
