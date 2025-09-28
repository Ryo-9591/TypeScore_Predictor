"""
スタイル設定モジュール
UIコンポーネント用のスタイル設定を提供
"""

# ダークテーマの共通スタイル設定
DARK_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#ffffff", "size": 12},
    "title": {"font": {"color": "#ffffff", "size": 16}, "x": 0.5},
    "xaxis": {
        "gridcolor": "#444",
        "linecolor": "#666",
        "tickcolor": "#666",
        "title_font": {"color": "#ffffff", "size": 14},
    },
    "yaxis": {
        "gridcolor": "#444",
        "linecolor": "#666",
        "tickcolor": "#666",
        "title_font": {"color": "#ffffff", "size": 14},
    },
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#ffffff"}},
}

# カラーパレット（ダークテーマに適した色）
COLORS = {
    "primary": "#007bff",
    "secondary": "#6c757d",
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "accent_blue": "#007bff",
    "accent_green": "#28a745",
    "accent_red": "#dc3545",
    "accent_yellow": "#ffc107",
    "accent_purple": "#6f42c1",
    "accent_pink": "#e83e8c",
    "accent_cyan": "#17a2b8",
}

# グラデーション色
GRADIENT_COLORS = [
    "#007bff",
    "#0056b3",
    "#004085",
    "#003d82",
    "#28a745",
    "#1e7e34",
    "#155724",
    "#0c5460",
    "#dc3545",
    "#c82333",
    "#bd2130",
    "#a71e2a",
    "#ffc107",
    "#e0a800",
    "#d39e00",
    "#c69500",
]

# レイアウトスタイル
LAYOUT_STYLES = {
    "main_container": {
        "backgroundColor": "#1a1a1a",
        "height": "100vh",
        "padding": "10px",
        "overflow": "hidden",
        "display": "flex",
        "flexDirection": "column",
        "boxSizing": "border-box",
    },
    "header": {
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "15px",
        "padding": "0 5px",
        "flex": "0 0 auto",
    },
    "stats_container": {
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr 1fr 1fr",
        "gridTemplateRows": "1fr",
        "gap": "10px",
        "marginBottom": "15px",
        "flex": "0 0 auto",
    },
    "user_container": {
        "backgroundColor": "#2d2d2d",
        "borderRadius": "8px",
        "padding": "12px",
        "flex": "0 0 30%",
        "minWidth": "300px",
        "maxWidth": "30%",
        "overflow": "hidden",
        "display": "flex",
        "flexDirection": "column",
    },
    "analysis_container": {
        "flex": "1",
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "minWidth": "0",
    },
    "analysis_panel": {
        "backgroundColor": "#2d2d2d",
        "borderRadius": "8px",
        "padding": "12px",
        "flex": "1",
        "overflow": "hidden",
        "boxSizing": "border-box",
    },
    "bottom_container": {
        "display": "flex",
        "gap": "10px",
        "flex": "1",
        "minHeight": "0",
    },
}

# コンポーネントスタイル
COMPONENT_STYLES = {
    "stats_card": {
        "backgroundColor": "#2d2d2d",
        "border": "1px solid #444",
        "borderRadius": "12px",
        "padding": "15px",
        "textAlign": "center",
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
    },
    "dropdown": {
        "backgroundColor": "#3d3d3d",
        "color": "#ffffff",
        "border": "1px solid #555",
        "borderRadius": "8px",
    },
    "input": {
        "backgroundColor": "#3d3d3d",
        "color": "#ffffff",
        "border": "1px solid #555",
        "borderRadius": "4px",
        "padding": "8px",
    },
    "button": {
        "backgroundColor": "#007bff",
        "color": "#ffffff",
        "border": "none",
        "borderRadius": "4px",
        "padding": "10px",
        "cursor": "pointer",
    },
}

# CSSスタイル文字列
CSS_STYLES = """
body {
    background-color: #0f1419;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

/* グローバル統計カードのスタイル */
.stats-card {
    background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.stats-card h3 {
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 500;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stats-card h2 {
    margin: 0 0 6px 0;
    font-size: 28px;
    font-weight: 700;
}

.stats-card p {
    margin: 0;
    font-size: 12px;
    color: #cccccc;
    opacity: 0.8;
}

/* パネルのスタイル */
.panel {
    background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* ヘッダーのスタイル */
.dashboard-header {
    background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

/* Plotly グラフのダークテーマ */
.js-plotly-plot .plotly .modebar {
    background-color: #2d2d2d !important;
    border-radius: 8px !important;
}
.js-plotly-plot .plotly .modebar-btn {
    color: #ffffff !important;
}
.js-plotly-plot .plotly .modebar-btn:hover {
    background-color: #444 !important;
}

/* ドロップダウンのダークテーマ */
.Select-control {
    background-color: #3d3d3d !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.Select-control:hover {
    border-color: #777 !important;
}
.Select-menu-outer {
    background-color: #3d3d3d !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
}
.Select-option {
    background-color: #3d3d3d !important;
    color: #ffffff !important;
    padding: 10px 15px !important;
}
.Select-option.is-focused {
    background-color: #555 !important;
}
.Select-option.is-selected {
    background-color: #007bff !important;
}
.Select-placeholder {
    color: #cccccc !important;
}
.Select-input {
    color: #ffffff !important;
}

/* アニメーション */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* スクロールバーのスタイル */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #2d2d2d;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #777;
}
"""


def get_theme_config():
    """テーマ設定を取得"""
    return DARK_THEME


def get_colors():
    """カラーパレットを取得"""
    return COLORS


def get_layout_styles():
    """レイアウトスタイルを取得"""
    return LAYOUT_STYLES


def get_component_styles():
    """コンポーネントスタイルを取得"""
    return COMPONENT_STYLES


def get_css_styles():
    """CSSスタイル文字列を取得"""
    return CSS_STYLES
