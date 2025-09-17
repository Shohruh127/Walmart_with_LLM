import pandas as pd
import joblib
import os
import json
from prophet import Prophet
import numpy as np

ARTI_DIR = "models/artifacts"
CSV_PATH = "data/Walmart_Sales.csv"

def to_serializable(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

def train_prophet_store(store_id: int, df: pd.DataFrame):
    store_df = df[df["Store"] == store_id][["Date", "Weekly_Sales"]].copy()
    store_df = store_df.rename(columns={"Date": "ds", "Weekly_Sales": "y"}).dropna()

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(store_df)

    # Save model
    path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}.pkl")
    joblib.dump(m, path)

    # Meta info
    meta = {
        "store": int(store_id),
        "n_obs": int(len(store_df)),
        "first_date": str(store_df["ds"].min().date()),
        "last_date": str(store_df["ds"].max().date()),
    }
    meta_path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=to_serializable)

    return {"store": store_id, "status": "ok"}

def main():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    os.makedirs(ARTI_DIR, exist_ok=True)

    results = []
    for s in sorted(df["Store"].unique()):
        try:
            res = train_prophet_store(s, df)
            print(res)
            results.append(res)
        except Exception as e:
            print({"store": s, "status": "fail", "error": str(e)})

    print("Saved per-store Prophet models & meta to:", ARTI_DIR)

if __name__ == "__main__":
    main()
