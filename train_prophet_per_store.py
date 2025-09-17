# train_prophet_per_store.py
import os, json
import pandas as pd
import joblib
from prophet import Prophet

CSV_PATH = "data/Walmart_Sales.csv"
ARTI_DIR = "models/artifacts"
os.makedirs(ARTI_DIR, exist_ok=True)

def train_prophet_store(store_id: int, df: pd.DataFrame):
    sdf = df[df["Store"] == store_id].copy()
    if len(sdf) < 60:
        return {"store": store_id, "status": "skipped (too short)"}

    s = sdf[["Date","Weekly_Sales","Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment"]].copy()
    s = s.dropna().sort_values("Date")
    s = s.rename(columns={"Date":"ds", "Weekly_Sales":"y"})

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    # optional regressors (Prophet will extrapolate them as constants → keep only seasonality if you prefer)
    for reg in ["Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment"]:
        m.add_regressor(reg)

    m.fit(s)
    path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}.pkl")
    joblib.dump(m, path)

    meta = {"store": store_id, "n_samples": len(s), "model_path": path}
    with open(os.path.join(ARTI_DIR, f"prophet_store_{store_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"store": store_id, "status": "ok", "path": path}

def main():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values(["Store","Date"]).reset_index(drop=True)

    results = []
    for s in sorted(df["Store"].unique()):
        res = train_prophet_store(s, df)
        results.append(res)
        print(res)

if __name__ == "__main__":
    main()
