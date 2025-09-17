import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Walmart Sales Forecast", layout="centered")
st.title("🧠🔮 Walmart Weekly Sales Forecast")

with st.form("query"):
    cols = st.columns(3)
    store_id = cols[0].number_input("Store ID", min_value=1, max_value=45, value=1, step=1)
    horizon = cols[1].number_input("Horizon", min_value=1, max_value=52, value=8, step=1)
    unit = cols[2].selectbox("Unit", ["weeks","months"], index=0)

    run_xgb = st.form_submit_button("Forecast with XGBoost")
    run_prophet = st.form_submit_button("Forecast with Prophet")

if run_xgb or run_prophet:
    model_name = "xgb" if run_xgb else "prophet"
    params = {
        "store_id": int(store_id),
        "horizon": int(horizon),
        "unit": unit,
        "model_name": model_name
    }
    try:
        r = requests.get("http://localhost:8000/forecast", params=params, timeout=30)
        data = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if not data.get("ok"):
        st.error(data.get("message", "Unknown error"))
    else:
        st.success(data["message"])
        df = pd.DataFrame(data["results"])
        st.dataframe(df, use_container_width=True)
        if "Pred_Weekly_Sales" in df.columns:
            st.line_chart(df.set_index("Date")["Pred_Weekly_Sales"])
