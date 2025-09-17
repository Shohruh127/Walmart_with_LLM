# utils/features.py
import pandas as pd

HOLIDAY_DUMMY_PREFIX = "Holiday_Type_"

BASE_FEATURES = [
    "Store","Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment",
    "Sales_lag_1","Sales_lag_2","Sales_lag_52",
    "Rolling_mean_4","Rolling_std_4","Rolling_mean_12",
    "Days_Since_Prev_Holiday","Days_To_Next_Holiday","Pre_Holiday_1w","Post_Holiday_1w"
]

def add_lag_rolling_per_store(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["Store","Date"])
    g = out.groupby("Store")["Weekly_Sales"]
    out["Sales_lag_1"]  = g.shift(1)
    out["Sales_lag_2"]  = g.shift(2)
    out["Sales_lag_52"] = g.shift(52)
    out["Rolling_mean_4"]  = g.shift(1).rolling(4).mean()
    out["Rolling_std_4"]   = g.shift(1).rolling(4).std()
    out["Rolling_mean_12"] = g.shift(1).rolling(12).mean()
    return out

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    holiday_dummies = [c for c in df.columns if c.startswith(HOLIDAY_DUMMY_PREFIX)]
    return BASE_FEATURES + holiday_dummies
