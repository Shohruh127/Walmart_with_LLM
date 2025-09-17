import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/forecast"  # change if deployed

st.set_page_config(page_title="Walmart Sales Forecast", layout="centered")
st.title("🧠🔮 Walmart Weekly Sales Forecast")

with st.form("query"):
    c1, c2, c3 = st.columns(3)
    store_id = c1.number_input("Store ID", min_value=1, max_value=45, value=1, step=1)
    horizon  = c2.number_input("Horizon", min_value=1, max_value=52, value=8, step=1)
    unit     = c3.selectbox("Unit", ["weeks", "months"], index=0)

    # 🔽 NEW: model selector (auto is default)
    model_choice = st.selectbox("Model", ["auto", "xgb", "prophet"], index=0,
                                help="Use 'auto' to select the best model per store based on training summary.")

    submitted = st.form_submit_button("Get Forecast")

if submitted:
    params = {
        "store_id": int(store_id),
        "horizon": int(horizon),
        "unit": unit,
        "model_name": model_choice,   # 🔽 pass the user's choice to the API
    }

    try:
        r = requests.get(API_URL, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if not data.get("ok"):
        st.error(data.get("message", "Unknown error"))
    else:
        # Show message from API (includes horizon cap note)
        st.success(data["message"])

        # Emphasize chosen model from the API response (could be auto-resolved)
        st.info(f"Model used: **{data.get('model_name','?').upper()}**")

        df = pd.DataFrame(data["results"])
        st.dataframe(df, use_container_width=True)

        if "Pred_Weekly_Sales" in df.columns:
            st.line_chart(df.set_index("Date")["Pred_Weekly_Sales"])
