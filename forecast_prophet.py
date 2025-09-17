# forecast_prophet.py
import joblib
import pandas as pd

ARTI_DIR = "models/artifacts"

def forecast_store_prophet(store_id: int, horizon_weeks: int = 8) -> pd.DataFrame:
    m = joblib.load(f"{ARTI_DIR}/prophet_store_{store_id}.pkl")
    # weekly frequency; Prophet uses 'ds' so we make a future frame
    future = m.make_future_dataframe(periods=horizon_weeks, freq="W-FRI")  # Kaggle Fridays
    fc = m.predict(future).tail(horizon_weeks)[["ds","yhat","yhat_lower","yhat_upper"]]
    fc = fc.rename(columns={"ds":"Date","yhat":"Pred_Weekly_Sales",
                            "yhat_lower":"lo","yhat_upper":"hi"})
    return fc

if __name__ == "__main__":
    print(forecast_store_prophet(1, 12).head())
