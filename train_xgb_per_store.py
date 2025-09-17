# train_xgb_per_store.py
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

CSV_PATH = "data/Walmart_Sales.csv"
ARTI_DIR = "models/artifacts"
os.makedirs(ARTI_DIR, exist_ok=True)

def train_one_store(store_id: int, df: pd.DataFrame) -> dict:
    sdf = df[df["Store"] == store_id].copy()
    if len(sdf) < 110:
        return {"store": store_id, "status": "skipped (too short)"}

    # Feature engineering
    sdf = add_holiday_features(sdf)
    sdf = add_lag_rolling_per_store(sdf)
    sdf = sdf.sort_values("Date").dropna().reset_index(drop=True)

    feat_cols = get_feature_columns(sdf)
    X = sdf[feat_cols]
    y = sdf["Weekly_Sales"]

    # time-aware split (last 20% test)
    split = int(len(sdf)*0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric="mae"
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False, early_stopping_rounds=50)

    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2  = float(r2_score(yte, pred))

    # Save model & metadata
    model_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}.json")
    model.save_model(model_path)  # json format (portable)
    meta = {
        "store": store_id,
        "n_samples": len(sdf),
        "train_end_date": str(sdf.loc[split-1, "Date"].date()),
        "test_start_date": str(sdf.loc[split, "Date"].date()),
        "features": feat_cols,
        "metrics": {"mae": mae, "r2": r2},
        "model_path": model_path
    }
    with open(os.path.join(ARTI_DIR, f"xgb_store_{store_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"store": store_id, "status": "ok", "MAE": mae, "R2": r2}

def main():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values(["Store","Date"]).reset_index(drop=True)

    stores = sorted(df["Store"].unique().tolist())
    results = []
    for s in stores:
        res = train_one_store(s, df)
        results.append(res)
        print(res)

    pd.DataFrame(results).to_csv(os.path.join(ARTI_DIR, "xgb_training_summary.csv"), index=False)
    print("\nSaved per-store models & summary to:", ARTI_DIR)

if __name__ == "__main__":
    main()
