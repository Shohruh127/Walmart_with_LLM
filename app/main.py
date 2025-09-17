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

# --------- Helpers ----------
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

# 🔹 Load hybrid summary at startup
HYBRID_SUMMARY = pd.read_csv(os.path.join(ARTI_DIR, "hybrid_training_summary.csv"))

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
    # ... your existing code ...
    # (no change needed here)
    return pd.DataFrame(out_rows)

def forecast_store_prophet(store_id: int, horizon_weeks: int) -> pd.DataFrame:
    # ... your existing code ...
    return fc

# --------- API schemas ----------
class ForecastResponse(BaseModel):
    ok: bool
    message: str
    store_id: int | None = None
    horizon_weeks: int | None = None
    model_name: Literal["xgb","prophet"] | None = None
    results: list[dict] | None = None

# --------- API endpoint with guardrails ----------
@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_id: int = Query(..., ge=1, le=45, description="Store ID (1..45)"),
    horizon: int = Query(8, ge=1, description="Forecast horizon"),
    unit: Literal["weeks","months"] = Query("weeks"),
    model_name: Literal["xgb","prophet","auto"] = Query("auto")   # 🔹 allow "auto"
):
    # Guardrail: cap horizon
    horizon_weeks = horizon if unit == "weeks" else horizon * 4
    capped = False
    if horizon_weeks > MAX_WEEKS:
        horizon_weeks = MAX_WEEKS
        capped = True

    # 🔹 Auto model selection
    if model_name == "auto":
        row = HYBRID_SUMMARY.loc[HYBRID_SUMMARY.store == store_id].iloc[0]
        model_name = row["Chosen_Model"]

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
