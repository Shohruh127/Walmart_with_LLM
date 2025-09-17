import numpy as np
import pandas as pd

HOLIDAY_STRS = [
    "2010-02-12","2010-09-10","2010-11-26","2010-12-31",
    "2011-02-11","2011-09-09","2011-11-25","2011-12-30",
    "2012-02-10","2012-09-07","2012-11-23","2012-12-28"
]
HOLIDAY_TYPES = {
    "2010-02-12":"SuperBowl","2010-09-10":"LaborDay","2010-11-26":"Thanksgiving","2010-12-31":"Christmas",
    "2011-02-11":"SuperBowl","2011-09-09":"LaborDay","2011-11-25":"Thanksgiving","2011-12-30":"Christmas",
    "2012-02-10":"SuperBowl","2012-09-07":"LaborDay","2012-11-23":"Thanksgiving","2012-12-28":"Christmas"
}

HOLIDAYS = pd.to_datetime(HOLIDAY_STRS)
HOLIDAYS_SORTED = np.sort(HOLIDAYS.values)
HOLIDAY_TYPE_MAP = {pd.to_datetime(k): v for k, v in HOLIDAY_TYPES.items()}

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Holiday_Type"] = out["Date"].map(HOLIDAY_TYPE_MAP).fillna("None")

    pos = np.searchsorted(HOLIDAYS_SORTED, out["Date"].values)
    prev_holiday = np.where(pos > 0, HOLIDAYS_SORTED[pos - 1], np.datetime64('NaT'))
    next_holiday = np.where(pos < len(HOLIDAYS_SORTED), HOLIDAYS_SORTED[pos], np.datetime64('NaT'))

    days_since_prev = (out["Date"].values - prev_holiday).astype('timedelta64[D]').astype('float')
    days_to_next   = (next_holiday - out["Date"].values).astype('timedelta64[D]').astype('float')

    days_since_prev = np.where(np.isnan(days_since_prev), 9999.0, days_since_prev)
    days_to_next    = np.where(np.isnan(days_to_next),    9999.0, days_to_next)

    out["Days_Since_Prev_Holiday"] = days_since_prev
    out["Days_To_Next_Holiday"]    = days_to_next
    out["Pre_Holiday_1w"]          = (out["Days_To_Next_Holiday"].between(1, 7)).astype(int)
    out["Post_Holiday_1w"]         = (out["Days_Since_Prev_Holiday"].between(1, 7)).astype(int)

    out = pd.get_dummies(out, columns=["Holiday_Type"], drop_first=True)
    return out
