from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import json
from xgboost import XGBRegressor
import joblib
import os

# ---- Config ----
ARTI_DIR = "models/artifacts"
CSV_PATH = "data/Walmart_Sales.csv"
MAX_WEEKS = 20  # guardrail: cap horizon (≈ 5 months)

app = FastAPI(title="Walmart Forecast API")

# --------- Helpers (mirror your training setup) ----------
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

def _load_xgb(store_id: int) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(os.path.join(ARTI_DIR, f"xgb_store_{store_id}.json"))
    return model

def _xgb_features_list(store_id: int) -> list[str]:
    meta_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["features"]

def forecast_store_xgb(store_id: int, horizon_weeks: int) -> pd.DataFrame:
    model = _load_xgb(store_id)
    feat_cols = _xgb_features_list(store_id)

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    cur = df[df["Store"] == store_id].copy().sort_values("Date")

    out_rows = []
    for _ in range(horizon_weeks):
        tmp = add_holiday_features(cur)
        tmp = add_lag_rolling_per_store(tmp).sort_values("Date")
        last_row = tmp.dropna().iloc[-1].copy()

        X = last_row[feat_cols].values.reshape(1, -1)
        pred = float(model.predict(X)[0])
        next_date = last_row["Date"] + pd.Timedelta(weeks=1)

        out_rows.append({"Date": str(next_date.date()), "Pred_Weekly_Sales": pred})

        # append prediction: hold last exogenous (simple baseline)
        cur = pd.concat([cur, pd.DataFrame([{
            "Store": store_id,
            "Date": next_date,
            "Weekly_Sales": pred,
            "Holiday_Flag": last_row.get("Holiday_Flag", 0),
            "Temperature": last_row.get("Temperature", cur["Temperature"].iloc[-1]),
            "Fuel_Price": last_row.get("Fuel_Price", cur["Fuel_Price"].iloc[-1]),
            "CPI": last_row.get("CPI", cur["CPI"].iloc[-1]),
            "Unemployment": last_row.get("Unemployment", cur["Unemployment"].iloc[-1]),
        }])], ignore_index=True)

    return pd.DataFrame(out_rows)

def forecast_store_prophet(store_id: int, horizon_weeks: int) -> pd.DataFrame:
    # Prophet per-store model is optional; train via train_prophet_per_store.py
    path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prophet model not found for store {store_id}.")
    m = joblib.load(path)
    future = m.make_future_dataframe(periods=horizon_weeks, freq="W-FRI")
    fc = m.predict(future).tail(horizon_weeks)[["ds","yhat","yhat_lower","yhat_upper"]]
    fc = fc.rename(columns={"ds":"Date","yhat":"Pred_Weekly_Sales","yhat_lower":"lo","yhat_upper":"hi"})
    fc["Date"] = fc["Date"].dt.date.astype(str)
    return fc

# --------- API schemas ----------
class ForecastResponse(BaseModel):
    ok: bool
    message: str
    store_id: int | None = None
    horizon_weeks: int | None = None
    model_name: Literal["xgb","prophet"] | None = None
    results: list[dict] | None = None  # [{Date, Pred_Weekly_Sales, ...}]

# --------- API endpoint with guardrails ----------
@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_id: int = Query(..., ge=1, le=45, description="Store ID (1..45)"),
    horizon: int = Query(8, ge=1, description="Forecast horizon"),
    unit: Literal["weeks","months"] = Query("weeks"),
    model_name: Literal["xgb","prophet"] = Query("xgb")
):
    # Guardrail: cap horizon
    horizon_weeks = horizon if unit == "weeks" else horizon * 4
    capped = False
    if horizon_weeks > MAX_WEEKS:
        horizon_weeks = MAX_WEEKS
        capped = True

    try:
        if model_name == "xgb":
            df_fc = forecast_store_xgb(store_id, horizon_weeks)
        else:
            df_fc = forecast_store_prophet(store_id, horizon_weeks)
    except FileNotFoundError as e:
        return ForecastResponse(ok=False, message=str(e))

    msg = f"Forecast for store {store_id} for next {horizon_weeks} weeks using {model_name.upper()}."
    if capped:
        msg += f" Requested horizon was capped to {MAX_WEEKS} weeks."

    return ForecastResponse(
        ok=True,
        message=msg,
        store_id=store_id,
        horizon_weeks=horizon_weeks,
        model_name=model_name,
        results=df_fc.to_dict(orient="records")
    )
