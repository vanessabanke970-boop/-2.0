# ============================================================
# main.py - 主程序入口
# 小红书用户数据分析：EDA + 可视化 + 机器学习 + SQL Server 存储
# ============================================================

import os
import sys
import logging
import numpy as np
import pandas as pd

from config      import DATA_CSV_PATH, USE_MOCK_DATA, OUTPUT_DIR
from visualizer  import run_all_visualizations, plot_ml_results
from ml_trainer  import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analysis.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       小红书用户消费行为分析系统  v1.0                     ║
║  EDA 可视化 ▸ 机器学习 ▸ SQL Server 数据存储              ║
╚══════════════════════════════════════════════════════════╝
"""


# ── 数据加载 ─────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not USE_MOCK_DATA and os.path.exists(DATA_CSV_PATH):
        logger.info(f"📂 从 CSV 加载数据: {DATA_CSV_PATH}")
        df = pd.read_csv(DATA_CSV_PATH)
    else:
        logger.info("🔧 使用模拟数据（29,452 条）")
        df = _generate_mock_data(29452)
    logger.info(f"✅ 数据加载完成，共 {len(df)} 条，{df.shape[1]} 列")
    return df


def _generate_mock_data(n: int = 29452, seed: int = 42) -> pd.DataFrame:
    """生成与科赛平台数据集结构一致的模拟数据"""
    rng = np.random.default_rng(seed)
    gender     = rng.choice([0, 1], size=n, p=[0.62, 0.38])
    # 女性平均消费略高
    revenue    = np.where(
        gender == 0,
        rng.lognormal(mean=4.8, sigma=1.1, size=n),
        rng.lognormal(mean=4.5, sigma=1.2, size=n),
    ).round(2)
    third_party = rng.integers(0, 8, size=n)
    engaged     = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    lifecycle   = rng.choice(["A", "B", "C"], size=n, p=[0.30, 0.45, 0.25])
    days_since  = rng.integers(1, 365, size=n)

    return pd.DataFrame({
        "Gender":              gender,
        "Revenue":             revenue,
        "3rd_party_stores":    third_party,
        "Engaged_last_30":     engaged,
        "Lifecycle":           lifecycle,
        "days_since_last_order": days_since,
    })


# ── 数据清洗 ─────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧹 开始数据清洗...")
    original_len = len(df)

    df = df.dropna()
    df = df[df["Revenue"] >= 0]
    df = df[df["Gender"].isin([0, 1])]
    df["Lifecycle"] = df["Lifecycle"].str.strip().str.upper()

    # 异常值剔除（IQR）
    Q1, Q3 = df["Revenue"].quantile(0.01), df["Revenue"].quantile(0.99)
    df = df[(df["Revenue"] >= Q1) & (df["Revenue"] <= Q3)]

    logger.info(f"✅ 清洗完成：{original_len} → {len(df)} 条（去除 {original_len - len(df)} 条异常）")
    return df.reset_index(drop=True)


# ── 统计摘要 ─────────────────────────────────────────────────
def compute_summary(df: pd.DataFrame) -> list[dict]:
    male_df   = df[df["Gender"] == 1]
    female_df = df[df["Gender"] == 0]

    summary = [
        {"metric_name": "total_users",          "metric_value": len(df),                        "category": "overview"},
        {"metric_name": "male_users",            "metric_value": len(male_df),                   "category": "gender"},
        {"metric_name": "female_users",          "metric_value": len(female_df),                 "category": "gender"},
        {"metric_name": "male_ratio",            "metric_value": round(len(male_df) / len(df), 4), "category": "gender"},
        {"metric_name": "avg_revenue_total",     "metric_value": round(df["Revenue"].mean(), 2), "category": "revenue"},
        {"metric_name": "avg_revenue_male",      "metric_value": round(male_df["Revenue"].mean(), 2), "category": "revenue"},
        {"metric_name": "avg_revenue_female",    "metric_value": round(female_df["Revenue"].mean(), 2), "category": "revenue"},
        {"metric_name": "median_revenue",        "metric_value": round(df["Revenue"].median(), 2), "category": "revenue"},
        {"metric_name": "avg_days_since_order",  "metric_value": round(df["days_since_last_order"].mean(), 2), "category": "behavior"},
        {"metric_name": "active_user_ratio",     "metric_value": round(df["Engaged_last_30"].mean(), 4), "category": "behavior"},
    ]

    print("\n📋 统计摘要：")
    for s in summary:
        print(f"  {s['category']:10s} | {s['metric_name']:28s} = {s['metric_value']}")
    return summary


# ── SQL Server 存储（可选） ───────────────────────────────────
def try_save_to_db(df, summary, ml_results):
    """尝试连接 SQL Server 并保存数据；失败时打印提示，不中断流程"""
    try:
        from db_handler import (
            create_database_if_not_exists, get_engine,
            create_tables, save_raw_data, save_ml_results, save_analysis_summary,
        )
        logger.info("\n💾 连接 SQL Server...")
        create_database_if_not_exists()
        engine = get_engine()
        create_tables(engine)
        save_raw_data(df, engine)
        save_analysis_summary(summary, engine)
        save_ml_results(ml_results, engine)
        logger.info("✅ 所有数据已成功保存到 SQL Server！")
    except Exception as e:
        logger.warning(
            f"\n⚠️  SQL Server 连接失败（{e}）\n"
            "   数据分析结果已保存为本地文件，SQL 存储跳过。\n"
            "   请确认：\n"
            "   1. SQL Server 服务已启动\n"
            "   2. config.py 中的连接信息正确\n"
            "   3. 已安装 'ODBC Driver 17 for SQL Server'\n"
        )


# ── 主流程 ───────────────────────────────────────────────────
def main():
    print(BANNER)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载 & 清洗
    df = load_data()
    df = clean_data(df)

    # 2. 统计摘要
    summary = compute_summary(df)

    # 3. EDA 可视化
    run_all_visualizations(df, OUTPUT_DIR)

    # 4. 机器学习
    ml_results, feature_importances, conf_matrices = train_and_evaluate(df, OUTPUT_DIR)
    plot_ml_results(ml_results, feature_importances, conf_matrices, OUTPUT_DIR)

    # 5. 保存到 SQL Server（失败不影响其他步骤）
    try_save_to_db(df, summary, ml_results)

    # 6. 输出最终汇总
    print("\n" + "═" * 60)
    print("🏆 机器学习模型性能汇总：")
    print(f"  {'模型':<22} {'Accuracy':>9} {'F1':>9} {'ROC-AUC':>9}")
    print("  " + "─" * 54)
    best = max(ml_results, key=lambda r: r["roc_auc"])
    for r in sorted(ml_results, key=lambda r: r["roc_auc"], reverse=True):
        marker = " ⭐ 最优" if r["model_name"] == best["model_name"] else ""
        print(f"  {r['model_name']:<22} {r['accuracy']:>9.4f} {r['f1_score']:>9.4f} {r['roc_auc']:>9.4f}{marker}")
    print("═" * 60)
    print(f"\n📁 所有图表已保存至: ./{OUTPUT_DIR}/")
    print("🎉 分析完成！\n")


if __name__ == "__main__":
    main()
