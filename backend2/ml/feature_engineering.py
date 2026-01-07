import pandas as pd
import numpy as np

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    def safe_div(a, b):
        return a / (b + 1e-6)

    # 这些比值只有在列是数值时才有意义；如果是字符串会变 NaN
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    if "Total Cholesterol" in X.columns and "HDL" in X.columns:
        tc = to_num(X["Total Cholesterol"])
        hdl = to_num(X["HDL"])
        X["chol_hdl_ratio"] = safe_div(tc, hdl).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "LDL" in X.columns and "HDL" in X.columns:
        ldl = to_num(X["LDL"])
        hdl = to_num(X["HDL"])
        X["ldl_hdl_ratio"] = safe_div(ldl, hdl).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "Triglycerides" in X.columns and "HDL" in X.columns:
        tg = to_num(X["Triglycerides"])
        hdl = to_num(X["HDL"])
        X["tg_hdl_ratio"] = safe_div(tg, hdl).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 年龄*血压这种交互：如果血压是类别，也会变 NaN -> 0
    if "Age" in X.columns and "Blood Pressure" in X.columns:
        age = to_num(X["Age"])
        bp = to_num(X["Blood Pressure"])
        X["age_bp"] = (age * bp).fillna(0.0)

    # CRP + 同型半胱氨酸：如果是类别（Normal/High/Low），会变 NaN -> 0，不再报错
    if "CRP Level" in X.columns and "Homocysteine Level" in X.columns:
        crp = to_num(X["CRP Level"])
        hcy = to_num(X["Homocysteine Level"])
        X["crp_hcy_sum"] = (crp + hcy).fillna(0.0)

    return X
