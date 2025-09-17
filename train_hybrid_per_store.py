# train_hybrid_per_store.py
import os, json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Optional Prophet (only needed for fallback)
try:
    import joblib
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---- Paths ----
CSV_PATH = "data/Walmart_Sales.csv"
ARTI_DIR = "models/artifacts"
os.makedirs(ARTI_DIR, exist_ok=True)

# ---- Utils from your repo ----
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

# ---- Seasonality helpers (adds to your raw df) ----
def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["Date"].dt.month.astype(int)
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["dayofyear"] = d["Date"].dt.dayofyear.astype(int)
    d["year"] = d["Date"].dt.year.astype(int)
    return d

def _ensure_xgb_compat_params():
    # Reasonable defaults; keep it simple and stable across xgboost versions.
    return dict(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
    )

def _fit_xgb(Xtr, ytr, Xte, yte):
    params = _ensure_xgb_compat_params()
    model = XGBRegressor(**params)
    # Early stopping supported for XGBRegressor across versions
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False, early_stopping_rounds=50)
    return model

def _evaluate(model, Xte, yte):
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2  = float(r2_score(yte, pred))
    return mae, r2, pred

def _save_xgb(store_id: int, model, feat_cols: list[str], split_date: str, test_start: str, mae: float, r2: float):
    model_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}.json")
    model.save_model(model_path)
    meta = {
        "store": store_id,
        "model": "xgb",
        "features": feat_cols,
        "train_end_date": split_date,
        "test_start_date": test_start,
        "metrics": {"mae": mae, "r2": r2},
        "model_path": model_path
    }
    with open(os.path.join(ARTI_DIR, f"xgb_store_{store_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def _fit_prophet_store(store_id: int, sdf: pd.DataFrame):
    """Fit Prophet on a single store with weekly frequency. Uses only y~seasonality by default."""
    s = sdf[["Date","Weekly_Sales"]].dropna().sort_values("Date").rename(columns={"Date":"ds","Weekly_Sales":"y"})
    if len(s) < 60:
        return None, None
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    m.fit(s)
    path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}.pkl")
    joblib.dump(m, path)
    return m, path

def train_one_store(store_id: int, df_raw: pd.DataFrame, r2_threshold: float = 0.0) -> dict:
    sdf = df_raw[df_raw["Store"] == store_id].copy()
    if len(sdf) < 110:
        return {"store": store_id, "status": "skipped (too short)"}

    # 1) Base features: holiday + seasonality + lags/rolling
    sdf = add_holiday_features(sdf)
    sdf = add_seasonality(sdf)
    sdf = add_lag_rolling_per_store(sdf)
    sdf = sdf.sort_values("Date").dropna().reset_index(drop=True)

    feat_cols = get_feature_columns(sdf) + ["month","weekofyear","dayofyear","year"]
    X = sdf[feat_cols]
    y = sdf["Weekly_Sales"]

    # time-aware split (last 20% test)
    split = int(len(sdf)*0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    # 2) Train XGB first
    xgb_model = _fit_xgb(Xtr, ytr, Xte, yte)
    mae_xgb, r2_xgb, _ = _evaluate(xgb_model, Xte, yte)

    # Save XGB immediately
    _save_xgb(
        store_id,
        xgb_model,
        feat_cols,
        split_date=str(sdf.loc[split-1, "Date"].date()),
        test_start=str(sdf.loc[split, "Date"].date()),
        mae=mae_xgb,
        r2=r2_xgb,
    )

    # 3) If XGB underperforms, try Prophet fallback
    chosen_model = "xgb"
    mae_final, r2_final = mae_xgb, r2_xgb
    prophet_path = None

    if r2_xgb < r2_threshold and PROPHET_AVAILABLE:
        m_prophet, path = _fit_prophet_store(store_id, sdf)
        if m_prophet is not None:
            prophet_path = path
            # (Optional) Evaluate Prophet on the same test range for apples-to-apples
            # Build Prophet predictions only on test dates:
            test_dates = sdf["Date"].iloc[split:]
            future = pd.DataFrame({"ds": test_dates})
            fc = m_prophet.predict(future)[["ds","yhat"]].rename(columns={"ds":"Date","yhat":"Pred"})
            merged = pd.merge(fc, sdf[["Date","Weekly_Sales"]].iloc[split:], on="Date", how="inner")
            mae_p = float(mean_absolute_error(merged["Weekly_Sales"], merged["Pred"]))
            r2_p  = float(r2_score(merged["Weekly_Sales"], merged["Pred"]))

            # If Prophet is better, mark it chosen
            if r2_p > r2_xgb:
                chosen_model = "prophet"
                mae_final, r2_final = mae_p, r2_p

            # Save a small meta for Prophet
            meta = {
                "store": store_id,
                "model": "prophet",
                "train_end_date": str(sdf.loc[split-1, "Date"].date()),
                "test_start_date": str(sdf.loc[split, "Date"].date()),
                "metrics": {"mae": mae_p, "r2": r2_p},
                "model_path": prophet_path
            }
            with open(os.path.join(ARTI_DIR, f"prophet_store_{store_id}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    result = {
        "store": store_id,
        "status": "ok",
        "XGB_MAE": mae_xgb, "XGB_R2": r2_xgb,
        "Prophet_Path": prophet_path if prophet_path else "",
        "Chosen_Model": chosen_model,
        "Final_MAE": mae_final, "Final_R2": r2_final
    }
    return result

def main():
    # Load & parse
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values(["Store","Date"]).reset_index(drop=True)

    stores = sorted(df["Store"].unique().tolist())
    results = []
    for s in stores:
        res = train_one_store(s, df, r2_threshold=0.0)  # flag underperformers
        results.append(res)
        print(res)

    pd.DataFrame(results).to_csv(os.path.join(ARTI_DIR, "hybrid_training_summary.csv"), index=False)
    print("\nSaved models & hybrid summary to:", ARTI_DIR)
    if not PROPHET_AVAILABLE:
        print("Note: Prophet not installed; fallback skipped for all stores.")

if __name__ == "__main__":
    main()
