import pandas as pd
import numpy as np
import os
import json
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ARTI_DIR = "models/artifacts"
CSV_PATH = "data/Walmart_Sales.csv"

# ---- Helper for JSON-safe dumping ----
def to_serializable(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

def train_one_store(store_id: int, df: pd.DataFrame):
    store_df = df[df["Store"] == store_id].copy().sort_values("Date")
    y = store_df["Weekly_Sales"]
    X = store_df.drop(columns=["Weekly_Sales", "Date"])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    # Save model
    model_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}.json")
    model.save_model(model_path)

    # Save meta
    meta = {
        "store": int(store_id),
        "features": list(X.columns),
        "train_size": int(len(Xtr)),
        "test_size": int(len(Xte)),
        "MAE": float(mean_absolute_error(yte, model.predict(Xte))),
        "R2": float(r2_score(yte, model.predict(Xte))),
    }
    meta_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=to_serializable)

    return meta

def main():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    os.makedirs(ARTI_DIR, exist_ok=True)

    results = []
    for s in sorted(df["Store"].unique()):
        try:
            res = train_one_store(s, df)
            print(res)
            results.append(res)
        except Exception as e:
            print({"store": s, "status": "fail", "error": str(e)})

    print("Saved per-store models & summary to:", ARTI_DIR)

if __name__ == "__main__":
    main()
