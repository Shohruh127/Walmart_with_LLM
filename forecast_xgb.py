# forecast_xgb.py
import json
import pandas as pd
from xgboost import XGBRegressor
from utils.holidays import add_holiday_features
from utils.features import add_lag_rolling_per_store, get_feature_columns

CSV_PATH = "data/Walmart_Sales.csv"
ARTI_DIR = "models/artifacts"

def _load_model(store_id: int) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(f"{ARTI_DIR}/xgb_store_{store_id}.json")
    return model

def _load_features_list(store_id: int) -> list[str]:
    meta_path = f"{ARTI_DIR}/xgb_store_{store_id}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["features"]

def forecast_store_xgb(store_id: int, horizon_weeks: int = 8) -> pd.DataFrame:
    model = _load_model(store_id)
    feat_cols = _load_features_list(store_id)

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    cur = df[df["Store"] == store_id].copy().sort_values("Date")

    out_rows = []
    for _ in range(horizon_weeks):
        # Feature engineering on current history
        tmp = add_holiday_features(cur)
        tmp = add_lag_rolling_per_store(tmp)
        tmp = tmp.sort_values("Date")

        last_row = tmp.dropna().iloc[-1].copy()
        X = last_row[feat_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]

        next_date = last_row["Date"] + pd.Timedelta(weeks=1)
        out_rows.append({"Date": next_date, "Pred_Weekly_Sales": float(pred)})

        # append prediction as next observed point for recursive forecasting
        cur = pd.concat([cur, pd.DataFrame([{
            "Store": store_id,
            "Date": next_date,
            "Weekly_Sales": pred,
            # fill exogenous with last known values (simple & robust)
            "Holiday_Flag": last_row["Holiday_Flag"],
            "Temperature": last_row["Temperature"],
            "Fuel_Price": last_row["Fuel_Price"],
            "CPI": last_row["CPI"],
            "Unemployment": last_row["Unemployment"],
        }])], ignore_index=True)

    return pd.DataFrame(out_rows)

if __name__ == "__main__":
    df_fc = forecast_store_xgb(store_id=1, horizon_weeks=12)
    print(df_fc.head(10))
