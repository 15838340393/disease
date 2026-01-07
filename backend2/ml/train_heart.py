# import os
# import joblib
# import numpy as np
# import pandas as pd
#
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# )
# from sklearn.feature_selection import SelectFromModel
# from sklearn.calibration import CalibratedClassifierCV
#
# from lightgbm import LGBMClassifier
#
# from ml.feature_engineering import add_engineered_features
#
# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "datasets", "heart.csv")  # 你的文件名
# ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
# os.makedirs(ARTIFACT_DIR, exist_ok=True)
#
# # ---------- Target ----------
# TARGET_COL = "Heart Disease Status"
#
#
# def evaluate(y_true, y_pred, y_proba):
#     return {
#         "accuracy": float(accuracy_score(y_true, y_pred)),
#         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
#         "recall": float(recall_score(y_true, y_pred, zero_division=0)),
#         "f1": float(f1_score(y_true, y_pred, zero_division=0)),
#         "auc": float(roc_auc_score(y_true, y_proba)),
#     }
#
#
# def best_threshold(y_true, y_proba):
#     best_t, best_f1 = 0.5, -1.0
#     for t in np.linspace(0.05, 0.95, 19):
#         y_pred = (y_proba >= t).astype(int)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1 = float(f1)
#             best_t = float(t)
#     return best_t, best_f1
#
#
# def main():
#     if not os.path.exists(DATA_PATH):
#         raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
#
#     df = pd.read_csv(DATA_PATH).dropna().copy()
#
#     # -------- Label mapping (Yes/No -> 1/0) --------
#     y_raw = df[TARGET_COL].astype(str).str.strip().str.lower()
#     y = y_raw.map({"no": 0, "yes": 1})
#
#     print("=== Label value counts (after mapping) ===")
#     print(y.value_counts(dropna=False))
#     print("y null count:", int(y.isna().sum()))
#     if y.isna().any():
#         bad = df.loc[y.isna(), TARGET_COL].head(20).tolist()
#         raise ValueError(f"Label mapping failed. Unexpected values in '{TARGET_COL}': {bad}")
#
#     X = df.drop(columns=[TARGET_COL])
#
#     # -------- Split --------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     # 计算不平衡比例（可写进论文）
#     pos = int((y_train == 1).sum())
#     neg = int((y_train == 0).sum())
#     print(f"\nTrain positives={pos}, negatives={neg}, imbalance_ratio(neg/pos)={neg/(pos+1e-6):.3f}\n")
#
#     # -------- Build preprocessor based on "engineered" columns --------
#     # 注意：工程特征在 pipeline 里动态生成，因此这里的列类型判断要基于“加完特征”的数据
#     X_train_fe = add_engineered_features(X_train)
#     cat_cols = [c for c in X_train_fe.columns if X_train_fe[c].dtype == "object"]
#     num_cols = [c for c in X_train_fe.columns if c not in cat_cols]
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
#             ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
#         ],
#         remainder="drop",
#     )
#
#     # -------- Base LGBM (handles imbalance) --------
#     lgbm_base = LGBMClassifier(
#         n_estimators=1200,
#         learning_rate=0.03,
#         num_leaves=31,
#         random_state=42,
#         class_weight="balanced",
#         # 让结果更稳定一点（可选）
#         subsample=0.9,
#         colsample_bytree=0.9,
#     )
#
#     # -------- Feature selection (reduce noise) --------
#     # 用一个较轻量的 LGBM 当 selector，按重要性保留 “中位数以上”的特征
#     selector_estimator = LGBMClassifier(
#         n_estimators=400,
#         learning_rate=0.05,
#         num_leaves=31,
#         random_state=42,
#         class_weight="balanced",
#     )
#
#     # -------- Pipeline: FeatureEngineering -> Preprocess -> Select -> Model --------
#     base_pipe = Pipeline(steps=[
#         ("feat", FunctionTransformer(add_engineered_features, validate=False)),
#         ("prep", preprocessor),
#         ("select", SelectFromModel(selector_estimator, threshold="median")),
#         ("model", lgbm_base),
#     ])
#
#     # -------- Calibration (improve probability quality; sometimes helps AUC a bit too) --------
#     # isotonic 效果通常更好，但数据量不够时也可以改成 method="sigmoid"
#     calibrated = CalibratedClassifierCV(
#         estimator=base_pipe,
#         method="isotonic",
#         cv=3
#     )
#
#     # -------- Cross-validated AUC (论文强烈建议给出) --------
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     cv_auc = cross_val_score(calibrated, X, y, cv=cv, scoring="roc_auc")
#     print("=== 5-fold CV AUC ===")
#     print("AUC scores:", cv_auc)
#     print("Mean AUC:", float(cv_auc.mean()), "Std:", float(cv_auc.std()))
#     print()
#
#     # -------- Fit & Evaluate on holdout test set --------
#     calibrated.fit(X_train, y_train)
#     y_proba = calibrated.predict_proba(X_test)[:, 1]
#
#     # 默认阈值 0.5
#     y_pred_05 = (y_proba >= 0.5).astype(int)
#     metrics_05 = evaluate(y_test, y_pred_05, y_proba)
#
#     # 阈值搜索（用于系统风险分级）
#     t_best, f1_best = best_threshold(y_test.values, y_proba)
#     y_pred_best = (y_proba >= t_best).astype(int)
#     metrics_best = evaluate(y_test, y_pred_best, y_proba)
#
#     print("=== Calibrated LGBM (threshold=0.5) ===")
#     for k, v in metrics_05.items():
#         print(f"{k} = {v}")
#
#     print("\n=== Calibrated LGBM (best threshold) ===")
#     print("best_threshold =", t_best, "best_f1 =", f1_best)
#     for k, v in metrics_best.items():
#         print(f"{k} = {v}")
#
#     # -------- Save artifact --------
#     artifact_path = os.path.join(ARTIFACT_DIR, "heart_pipeline.joblib")
#     joblib.dump({
#         "pipeline": calibrated,              # 注意：保存校准后的整体对象
#         "feature_names": list(X.columns),    # 推理时前端只需提供这些“原始字段”
#         "target_col": TARGET_COL,
#         "threshold": t_best,                 # 系统判定阳性的阈值
#         "metrics_threshold_05": metrics_05,
#         "metrics_best_threshold": metrics_best,
#         "cv_auc_mean": float(cv_auc.mean()),
#         "cv_auc_std": float(cv_auc.std()),
#         "model_name": "calibrated_lgbm_fs"
#     }, artifact_path)
#
#     print("\nSaved to:", artifact_path)
#
#
# if __name__ == "__main__":
#     main()
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "heart.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL = "Heart Disease Status"

# 你CSV里除了label以外的全部字段（增强版）
ADV_FEATURES = [
    "Age",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level",
    "Exercise Habits",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "BMI",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Alcohol Consumption",
    "Stress Level",
    "Sleep Hours",
    "Sugar Consumption",
    "Triglyceride Level",
    "Fasting Blood Sugar",
    "CRP Level",
    "Homocysteine Level",
]

# 明确指定数值列（其余全部当类别）
NUM_COLS = ["Age", "BMI", "Sleep Hours"]
CAT_COLS = [c for c in ADV_FEATURES if c not in NUM_COLS]


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def best_threshold(y_true, y_proba):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def main():
    df = pd.read_csv(DATA_PATH).dropna().copy()

    y = df[TARGET_COL].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    if y.isna().any():
        bad = df.loc[y.isna(), TARGET_COL].head(20).tolist()
        raise ValueError(f"Label mapping failed. Unexpected values: {bad}")

    X = df[ADV_FEATURES].copy()

    # 类型强制：避免 Age/BMI/Sleep Hours 被读成字符串
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna().copy()
    y = y.loc[X.index]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = []
    candidates.append(("lr", LogisticRegression(max_iter=4000, class_weight="balanced")))
    candidates.append(("rf", RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced")))

    if LGBMClassifier is not None:
        candidates.append(("lgbm", LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.9
        )))

    best = None

    for name, model in candidates:
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred_05 = (y_proba >= 0.5).astype(int)
        metrics_05 = evaluate(y_test, y_pred_05, y_proba)

        t_best, f1_best = best_threshold(y_test.values, y_proba)
        y_pred_best = (y_proba >= t_best).astype(int)
        metrics_best = evaluate(y_test, y_pred_best, y_proba)

        print(f"\n=== {name} ===")
        print("threshold=0.5:", metrics_05)
        print("best_threshold:", t_best, "best_f1:", f1_best)
        print("metrics_best:", metrics_best)

        # 选最优：先 AUC，再 best_f1
        if best is None:
            best = (name, pipe, t_best, metrics_05, metrics_best)
        else:
            best_name, best_pipe, best_t, best_m05, best_mbest = best
            if (metrics_best["auc"] > best_mbest["auc"] + 1e-6) or (
                abs(metrics_best["auc"] - best_mbest["auc"]) <= 1e-6 and metrics_best["f1"] > best_mbest["f1"]
            ):
                best = (name, pipe, t_best, metrics_05, metrics_best)

    best_name, best_pipe, best_t, best_m05, best_mbest = best

    artifact_path = os.path.join(ARTIFACT_DIR, "heart_pipeline.joblib")
    joblib.dump({
        "pipeline": best_pipe,
        "feature_names": ADV_FEATURES,
        "target_col": TARGET_COL,
        "threshold": float(best_t),
        "metrics_threshold_05": best_m05,
        "metrics_best_threshold": best_mbest,
        "model_name": best_name,
        "mode": "advanced"
    }, artifact_path)

    print("\n✅ Saved to:", artifact_path)


if __name__ == "__main__":
    main()
