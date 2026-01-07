# ml/train_heart_basic.py
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


# ========== Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "heart.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ========== Target ==========
TARGET_COL = "Heart Disease Status"

# ========== Basic (practical) feature set ==========
# 只用“居家自查”可获得/可自述字段
BASIC_FEATURES = [
    "Age",
    "Gender",
    "Blood Pressure",
    "Exercise Habits",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "BMI",
    "High Blood Pressure",
    "Alcohol Consumption",
    "Stress Level",
    "Sleep Hours",
    "Sugar Consumption",
]


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
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).dropna().copy()

    # -------- 1) label mapping --------
    y_raw = df[TARGET_COL].astype(str).str.strip().str.lower()
    y = y_raw.map({"no": 0, "yes": 1})

    print("=== Label value counts (after mapping) ===")
    print(y.value_counts(dropna=False))
    print("y null count:", int(y.isna().sum()))
    if y.isna().any():
        bad = df.loc[y.isna(), TARGET_COL].head(20).tolist()
        raise ValueError(f"Label mapping failed. Unexpected values in '{TARGET_COL}': {bad}")

    # -------- 2) select basic features --------
    missing = [c for c in BASIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    X = df[BASIC_FEATURES].copy()

    # -------- 3) infer column types --------
    # 若某些列是字符串数值（例如 "120/80" 或 "High/Low"），先统一当作类别更稳
    num_cols = ["Age", "BMI", "Sleep Hours"]
    cat_cols = [c for c in BASIC_FEATURES if c not in num_cols]


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    # -------- 4) split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"\nTrain positives={pos}, negatives={neg}, imbalance_ratio(neg/pos)={neg/(pos+1e-6):.3f}\n")

    # -------- 5) candidates (basic) --------
    candidates = [
        ("lr", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced")),
    ]

    results = []
    best = None

    for name, model in candidates:
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred_05 = (y_proba >= 0.5).astype(int)
        metrics_05 = evaluate(y_test, y_pred_05, y_proba)

        t_best, f1_best = best_threshold(y_test.values, y_proba)
        y_pred_best = (y_proba >= t_best).astype(int)
        metrics_best = evaluate(y_test, y_pred_best, y_proba)

        print(f"=== {name} (threshold=0.5) ===")
        for k, v in metrics_05.items():
            print(f"{k} = {v}")

        print(f"\n=== {name} (best threshold) ===")
        print("best_threshold =", t_best, "best_f1 =", f1_best)
        for k, v in metrics_best.items():
            print(f"{k} = {v}")
        print()

        results.append((name, metrics_best["auc"], f1_best, t_best, pipe, metrics_05, metrics_best))

        # 选最优：先 AUC，若接近再看 best_f1
        if best is None:
            best = results[-1]
        else:
            _, best_auc, best_f1, *_ = best
            cur_auc = metrics_best["auc"]
            if (cur_auc > best_auc + 1e-6) or (abs(cur_auc - best_auc) <= 1e-6 and f1_best > best_f1):
                best = results[-1]

    best_name, best_auc, best_f1, best_t, best_pipe, best_metrics_05, best_metrics_best = best

    print("✅ Best BASIC model:", best_name)
    print("Best AUC:", best_auc, "Best F1:", best_f1, "Best threshold:", best_t)

    # -------- 6) save artifact --------
    artifact_path = os.path.join(ARTIFACT_DIR, "heart_basic_pipeline.joblib")
    joblib.dump({
        "pipeline": best_pipe,
        "feature_names": BASIC_FEATURES,          # 前端/后端按这 12 个字段传参即可
        "target_col": TARGET_COL,
        "threshold": best_t,
        "metrics_threshold_05": best_metrics_05,
        "metrics_best_threshold": best_metrics_best,
        "model_name": best_name,
        "mode": "basic"
    }, artifact_path)

    print("Saved to:", artifact_path)


if __name__ == "__main__":
    main()
