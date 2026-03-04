# ============================================================
# config.py - 项目配置文件
# ============================================================

# ── SQL Server 连接配置 ──────────────────────────────────────
DB_CONFIG = {
    "server":   "localhost",          # SQL Server 地址
    "database": "XiaohongshuDB",      # 数据库名（会自动创建）
    "driver":   "ODBC Driver 17 for SQL Server",
    # 使用 Windows 身份验证（Trusted_Connection）
    # 如需用户名密码，将下面两行取消注释并填写
    # "username": "sa",
    # "password": "your_password",
    "trusted_connection": True,
}

# ── 数据集配置 ───────────────────────────────────────────────
# 科赛平台数据集（本地 CSV 路径，下载后放在此处）
DATA_CSV_PATH = "data/xiaohongshu_users.csv"

# 若本地无 CSV，使用内置模拟数据
USE_MOCK_DATA = True   # ← 改为 False 后读取真实 CSV

# ── 输出目录 ────────────────────────────────────────────────
OUTPUT_DIR = "output"  # 图表和模型保存目录

# ── 机器学习配置 ─────────────────────────────────────────────
ML_CONFIG = {
    "test_size":      0.2,
    "random_state":   42,
    "cv_folds":       5,
}
