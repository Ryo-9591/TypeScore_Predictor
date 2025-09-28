from dash import dcc, html
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class UserSelector:
    """ユーザー選択コンポーネントクラス"""

    @staticmethod
    def create(users: List[str], selected_user: Optional[str] = None) -> html.Div:
        """
        ユーザー選択ドロップダウンを作成

        Args:
            users: ユーザーIDのリスト
            selected_user: 選択されたユーザーID

        Returns:
            ユーザー選択コンポーネントのDiv
        """
        user_options = [{"label": f"ユーザー {user}", "value": user} for user in users]

        return html.Div(
            [
                html.Label(
                    "ユーザー選択:",
                    style={
                        "color": "#ffffff",
                        "fontSize": "14px",
                        "marginBottom": "5px",
                    },
                ),
                dcc.Dropdown(
                    id="user-selector",
                    options=user_options,
                    value=selected_user,
                    style={
                        "backgroundColor": "#3d3d3d",
                        "color": "#ffffff",
                        "marginBottom": "20px",
                    },
                ),
            ]
        )
