# ============================================================
# db_handler.py - SQL Server 数据库操作模块
# ============================================================

import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
import urllib
import logging
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection_string() -> str:
    """生成 SQLAlchemy 连接字符串"""
    if DB_CONFIG.get("trusted_connection"):
        params = urllib.parse.quote_plus(
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"Trusted_Connection=yes;"
        )
    else:
        params = urllib.parse.quote_plus(
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
        )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def get_engine():
    """创建 SQLAlchemy Engine"""
    conn_str = get_connection_string()
    engine = create_engine(conn_str, fast_executemany=True)
    logger.info("✅ SQLAlchemy Engine 创建成功")
    return engine


def create_database_if_not_exists():
    """若数据库不存在则自动创建"""
    if DB_CONFIG.get("trusted_connection"):
        conn_str = (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE=master;"
            f"Trusted_Connection=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE=master;"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
        )
    try:
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        db_name = DB_CONFIG["database"]
        cursor.execute(f"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = '{db_name}') CREATE DATABASE [{db_name}]")
        conn.close()
        logger.info(f"✅ 数据库 [{db_name}] 已就绪")
    except Exception as e:
        logger.error(f"❌ 创建数据库失败: {e}")
        raise


def create_tables(engine):
    """创建所需数据表"""
    ddl = """
    -- 原始用户数据表
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_raw' AND xtype='U')
    CREATE TABLE user_raw (
        id               INT IDENTITY(1,1) PRIMARY KEY,
        gender           INT            NOT NULL,   -- 1=男, 0=女
        revenue          FLOAT          NOT NULL,   -- 下单金额
        third_party      INT            NOT NULL,   -- 第三方购买数
        engaged_last_30  INT            NOT NULL,   -- 近30天是否活跃
        lifecycle        VARCHAR(10)    NOT NULL,   -- 用户生命周期
        days_since_order INT            NOT NULL,   -- 距上次下单天数
        created_at       DATETIME       DEFAULT GETDATE()
    );

    -- 机器学习预测结果表
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ml_predictions' AND xtype='U')
    CREATE TABLE ml_predictions (
        id               INT IDENTITY(1,1) PRIMARY KEY,
        model_name       VARCHAR(50)    NOT NULL,
        accuracy         FLOAT          NOT NULL,
        precision_score  FLOAT          NOT NULL,
        recall_score     FLOAT          NOT NULL,
        f1_score         FLOAT          NOT NULL,
        roc_auc          FLOAT          NOT NULL,
        trained_at       DATETIME       DEFAULT GETDATE()
    );

    -- 统计摘要表
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='analysis_summary' AND xtype='U')
    CREATE TABLE analysis_summary (
        id               INT IDENTITY(1,1) PRIMARY KEY,
        metric_name      VARCHAR(100)   NOT NULL,
        metric_value     FLOAT          NOT NULL,
        category         VARCHAR(50),
        created_at       DATETIME       DEFAULT GETDATE()
    );
    """
    with engine.connect() as conn:
        for statement in ddl.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    logger.info("✅ 数据表创建完成")


def save_raw_data(df: pd.DataFrame, engine):
    """将清洗后的原始数据写入 user_raw 表"""
    df_to_save = df.rename(columns={
        "Gender":            "gender",
        "Revenue":           "revenue",
        "3rd_party_stores":  "third_party",
        "Engaged_last_30":   "engaged_last_30",
        "Lifecycle":         "lifecycle",
        "days_since_last_order": "days_since_order",
    })[["gender", "revenue", "third_party", "engaged_last_30", "lifecycle", "days_since_order"]]

    df_to_save.to_sql("user_raw", engine, if_exists="append", index=False)
    logger.info(f"✅ 已保存 {len(df_to_save)} 条原始数据到 user_raw")


def save_ml_results(results: dict, engine):
    """保存机器学习结果到 ml_predictions 表"""
    df_results = pd.DataFrame(results)
    df_results.to_sql("ml_predictions", engine, if_exists="append", index=False)
    logger.info("✅ 机器学习结果已保存到 ml_predictions")


def save_analysis_summary(summary: list[dict], engine):
    """保存统计摘要到 analysis_summary 表"""
    df_summary = pd.DataFrame(summary)
    df_summary.to_sql("analysis_summary", engine, if_exists="append", index=False)
    logger.info("✅ 统计摘要已保存到 analysis_summary")
