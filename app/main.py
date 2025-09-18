# app/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import json
import os
import joblib
from xgboost import XGBRegressor

# ---- Config ----
ARTI_DIR = "models/artifacts"
CSV_PATH = "data/Walmart_Sales.csv"
MAX_WEEKS = 20  # guardrail: cap horizon (≈ 5 months)

app = FastAPI(title="Walmart Forecast API")

# --------- Helpers ----------
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

# Try to load the hybrid summary once at startup; tolerate missing file
HYBRID_SUMMARY_PATH = os.path.join(ARTI_DIR, "hybrid_training_summary.csv")
if os.path.exists(HYBRID_SUMMARY_PATH):
    HYBRID_SUMMARY = pd.read_csv(HYBRID_SUMMARY_PATH)
else:
    # Empty frame with expected columns so lookups won't crash
    HYBRID_SUMMARY = pd.DataFrame(columns=["store", "Chosen_Model"])

# --------- Low-level loaders ----------
def _load_xgb(store_id: int) -> XGBRegressor:
    """Load a per-store XGBoost model saved with .save_model(json)."""
    path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"XGB model not found for store {store_id}: {path}")
    model = XGBRegressor()
    model.load_model(path)
    return model

def _xgb_features_list(store_id: int) -> list[str]:
    """Read the per-store feature list from the meta JSON."""
    meta_path = os.path.join(ARTI_DIR, f"xgb_store_{store_id}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"XGB meta not found for store {store_id}: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["features"]

def _load_prophet(store_id: int):
    """Load a per-store Prophet model."""
    path = os.path.join(ARTI_DIR, f"prophet_store_{store_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prophet model not found for store {store_id}: {path}")
    return joblib.load(path)

# --------- Forecast functions ----------
def forecast_store_xgb(store_id: int, horizon_weeks: int) -> pd.DataFrame:
    """
    Iterative weekly forecast using XGBoost with recursive strategy.
    Exogenous future values are held at last known values (simple baseline).
    """
    # Load model + feature list
    model = _load_xgb(store_id)
    feat_cols = _xgb_features_list(store_id)

    # Load store history
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    cur = df[df["Store"] == store_id].copy().sort_values("Date").reset_index(drop=True)

    if cur.empty:
        raise FileNotFoundError(f"No rows for store {store_id} found in {CSV_PATH}")

    out_rows = []
    for _ in range(horizon_weeks):
        # Build features on the current history
        tmp = add_holiday_features(cur)
        tmp = add_lag_rolling_per_store(tmp)
        tmp = tmp.sort_values("Date")

        tmp_non_na = tmp.dropna()
        if tmp_non_na.empty:
            raise RuntimeError("Not enough history to create lag/rolling features (all NA).")

        last_row = tmp_non_na.iloc[-1].copy()

        # Prepare feature vector
        missing_cols = [c for c in feat_cols if c not in tmp_non_na.columns]
        if missing_cols:
            raise RuntimeError(f"Missing expected feature columns: {missing_cols}")

        X = last_row[feat_cols].values.reshape(1, -1)
        pred = float(model.predict(X)[0])

        next_date = last_row["Date"] + pd.Timedelta(weeks=1)
        out_rows.append({"Date": str(next_date.date()), "Pred_Weekly_Sales": pred})

        # Append the prediction as the next observation.
        # Hold last known exogenous values (simple but robust baseline).
        # If any column is missing in cur, fill with np.nan so feature builder can handle it.
        def _get(col, fallback=np.nan):
            if col in last_row and pd.notna(last_row[col]):
                return last_row[col]
            if col in cur.columns and len(cur) > 0 and pd.notna(cur[col].iloc[-1]):
                return cur[col].iloc[-1]
            return fallback

        cur = pd.concat(
            [
                cur,
                pd.DataFrame(
                    [
                        {
                            "Store": store_id,
                            "Date": next_date,
                            "Weekly_Sales": pred,
                            "Holiday_Flag": _get("Holiday_Flag", 0),
                            "Temperature": _get("Temperature"),
                            "Fuel_Price": _get("Fuel_Price"),
                            "CPI": _get("CPI"),
                            "Unemployment": _get("Unemployment"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(out_rows)

def forecast_store_prophet(store_id: int, horizon_weeks: int) -> pd.DataFrame:
    """
    Weekly forecast using Prophet (model already trained per store).
    """
    m = _load_prophet(store_id)
    # Walmart Kaggle weeks are Fridays; using W-FRI keeps alignment
    future = m.make_future_dataframe(periods=horizon_weeks, freq="W-FRI")
    fc = m.predict(future).tail(horizon_weeks)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc = fc.rename(
        columns={
            "ds": "Date",
            "yhat": "Pred_Weekly_Sales",
            "yhat_lower": "lo",
            "yhat_upper": "hi",
        }
    )
    fc["Date"] = pd.to_datetime(fc["Date"]).dt.date.astype(str)
    return fc

# --------- API schemas ----------
class ForecastResponse(BaseModel):
    ok: bool
    message: str
    store_id: int | None = None
    horizon_weeks: int | None = None
    model_name: Literal["xgb", "prophet"] | None = None
    results: list[dict] | None = None

# --------- API endpoint with guardrails ----------
@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_id: int = Query(..., ge=1, le=45, description="Store ID (1..45)"),
    horizon: int = Query(8, ge=1, description="Forecast horizon"),
    unit: Literal["weeks", "months"] = Query("weeks"),
    model_name: Literal["xgb", "prophet", "auto"] = Query("auto"),
):
    # Guardrail: cap horizon
    horizon_weeks = horizon if unit == "weeks" else horizon * 4
    capped = False
    if horizon_weeks > MAX_WEEKS:
        horizon_weeks = MAX_WEEKS
        capped = True

    # Auto model selection from hybrid summary; default to XGB if not found
    chosen = model_name
    if model_name == "auto":
        row = HYBRID_SUMMARY.loc[HYBRID_SUMMARY["store"] == store_id]
        if row.empty:
            chosen = "xgb"  # default if no hybrid summary
        else:
            chosen = str(row.iloc[0]["Chosen_Model"]).lower()
            if chosen not in ("xgb", "prophet"):
                chosen = "xgb"

    # Run forecast
    try:
        if chosen == "xgb":
            df_fc = forecast_store_xgb(store_id, horizon_weeks)
        else:
            df_fc = forecast_store_prophet(store_id, horizon_weeks)
    except FileNotFoundError as e:
        return ForecastResponse(ok=False, message=str(e))
    except Exception as e:
        return ForecastResponse(ok=False, message=f"Forecast error: {e}")

    msg = f"Forecast for store {store_id} for next {horizon_weeks} weeks using {chosen.upper()}."
    if capped:
        msg += f" Requested horizon was capped to {MAX_WEEKS} weeks."

    return ForecastResponse(
        ok=True,
        message=msg,
        store_id=store_id,
        horizon_weeks=horizon_weeks,
        model_name=chosen,  # echoes the actually-used model
        results=df_fc.to_dict(orient="records"),
    )
