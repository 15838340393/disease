import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# 缓存，避免每次请求都从磁盘加载
_CACHE = {}

def _risk_level(p: float) -> str:
    # 你也可以按论文设定：<0.3 low, <0.7 medium, else high
    if p < 0.3:
        return "low"
    if p < 0.7:
        return "medium"
    return "high"

def load_artifact(filename: str):
    path = os.path.join(ARTIFACT_DIR, filename)
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]

def predict_from_artifact(artifact_filename: str, features: dict):
    obj = load_artifact(artifact_filename)

    pipe = obj["pipeline"]
    feature_names = obj["feature_names"]
    threshold = float(obj.get("threshold", 0.5))
    model_name = obj.get("model_name", "unknown")
    mode = obj.get("mode", "unknown")

    # 缺字段校验（前端没传就报错，避免 silent bug）
    missing = [c for c in feature_names if c not in features]
    if missing:
        return None, {
            "msg": f"missing required features: {missing[:10]}" + ("..." if len(missing) > 10 else ""),
            "missing": missing
        }

    X = pd.DataFrame([[features[c] for c in feature_names]], columns=feature_names)

    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)

    return {
        "probability": proba,
        "threshold": threshold,
        "pred_label": pred,
        "risk_level": _risk_level(proba),
        "model_name": model_name,
        "mode": mode,
        "metrics_best_threshold": obj.get("metrics_best_threshold"),
        "metrics_threshold_05": obj.get("metrics_threshold_05"),
    }, None
