import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/forecast"  # change if deployed

st.title("🛒 Walmart Sales Forecasting")

# User inputs
store_id = st.number_input("Select Store ID", min_value=1, max_value=45, value=1, step=1)
horizon = st.slider("Forecast horizon (weeks)", 1, 20, 8)

if st.button("Get Forecast"):
    params = {
        "store_id": store_id,
        "horizon": horizon,
        "unit": "weeks",
        "model_name": "auto",   # 🔹 let API auto-select best model
    }
    resp = requests.get(API_URL, params=params)
    data = resp.json()

    if not data["ok"]:
        st.error(data["message"])
    else:
        # 🔹 Show which model was chosen
        st.info(f"Model chosen for Store {store_id}: **{data['model_name'].upper()}**")

        # Display predictions as a table
        df = pd.DataFrame(data["results"])
        st.dataframe(df)

        # Plot line chart
        st.line_chart(df.set_index("Date")["Pred_Weekly_Sales"])
