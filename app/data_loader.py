import pandas as pd


class DataLoader:
    """データの読み込みを担当するクラス"""

    def __init__(self, data_path="data/"):
        """
        Args:
            data_path (str): データファイルのパス
        """
        self.data_path = data_path
        self.m_user = None
        self.t_miss = None
        self.t_score = None

    def load_data(self):
        """すべてのデータファイルを読み込む"""
        try:
            self.m_user = pd.read_csv(f"{self.data_path}m_user.csv")
            self.t_miss = pd.read_csv(f"{self.data_path}t_miss.csv")
            self.t_score = pd.read_csv(f"{self.data_path}t_score.csv")

            print("データの読み込みが完了しました")
            print(f"ユーザーデータ: {len(self.m_user)}件")
            print(f"ミスタイプデータ: {len(self.t_miss)}件")
            print(f"スコアデータ: {len(self.t_score)}件")

            return True
        except Exception as e:
            print(f"データの読み込みに失敗しました: {e}")
            return False

    def get_data(self):
        """読み込んだデータを返す"""
        return self.m_user, self.t_miss, self.t_score

    def get_user_mapping(self):
        """ユーザーIDとユーザー名のマッピングを返す"""
        if self.m_user is not None:
            return dict(zip(self.m_user["user_id"], self.m_user["username"]))
        return {}
