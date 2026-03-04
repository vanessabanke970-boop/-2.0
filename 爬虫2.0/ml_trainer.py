# ============================================================
# ml_trainer.py - 机器学习训练模块
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.metrics            import (accuracy_score, precision_score, recall_score,
                                        f1_score, roc_auc_score, confusion_matrix,
                                        classification_report, roc_curve)
from sklearn.pipeline           import Pipeline
import joblib
import logging
from config import ML_CONFIG

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame):
    """特征工程：编码 Lifecycle，分离 X/y"""
    df_ml = df.copy()

    # 编码分类变量
    le = LabelEncoder()
    df_ml["Lifecycle_encoded"] = le.fit_transform(df_ml["Lifecycle"])

    feature_cols = ["Revenue", "3rd_party_stores", "Engaged_last_30",
                    "days_since_last_order", "Lifecycle_encoded"]
    X = df_ml[feature_cols]
    y = df_ml["Gender"]

    return X, y, feature_cols, le


def build_pipelines() -> dict:
    """构建带标准化的模型 Pipeline"""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=ML_CONFIG["random_state"])),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                random_state=ML_CONFIG["random_state"], n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=ML_CONFIG["random_state"])),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, kernel="rbf",
                          random_state=ML_CONFIG["random_state"])),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame, output_dir: str):
    """训练所有模型，返回结果列表、特征重要性、混淆矩阵"""
    X, y, feature_cols, _ = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ML_CONFIG["test_size"],
        random_state=ML_CONFIG["random_state"],
        stratify=y,
    )

    pipelines         = build_pipelines()
    results           = []
    feature_importances = {}
    conf_matrices     = {}
    roc_data          = {}

    print("\n🤖 开始训练机器学习模型...\n")
    skf = StratifiedKFold(n_splits=ML_CONFIG["cv_folds"], shuffle=True,
                          random_state=ML_CONFIG["random_state"])

    for name, pipeline in pipelines.items():
        print(f"  ▶ 训练 {name}...")
        pipeline.fit(X_train, y_train)

        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

        acc       = accuracy_score(y_test, y_pred)
        prec      = precision_score(y_test, y_pred, zero_division=0)
        rec       = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        results.append({
            "model_name":      name,
            "accuracy":        round(acc,     4),
            "precision_score": round(prec,    4),
            "recall_score":    round(rec,     4),
            "f1_score":        round(f1,      4),
            "roc_auc":         round(roc_auc, 4),
        })

        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        roc_data[name]      = roc_curve(y_test, y_pred_prob)

        # 特征重要性（树模型）
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            feature_importances[name] = dict(zip(feature_cols, clf.feature_importances_))

        print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"CV-AUC={cv_scores.mean():.4f}(±{cv_scores.std():.4f})")
        print(classification_report(y_test, y_pred, target_names=["女性", "男性"]))

        # 保存模型
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"model_{name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"    💾 模型已保存: {model_path}")

    # 绘制 ROC 曲线
    _plot_roc_curves(roc_data, output_dir)

    print("\n✅ 所有模型训练完成！")
    return results, feature_importances, conf_matrices


def _plot_roc_curves(roc_data: dict, output_dir: str):
    """绘制所有模型 ROC 曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4A90D9", "#E8739A", "#5CB85C", "#F0AD4E"]

    for (name, (fpr, tpr, _)), color in zip(roc_data.items(), colors):
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC 曲线对比", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 已保存: {path}")
