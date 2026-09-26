import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("Predict future store sales using **LightGBM** with lag features and promo scenario simulation.")

API_URL = "http://127.0.0.1:8000"

# Sidebar Controls
st.sidebar.header("Forecast Parameters")
store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-08-01"))
horizon_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14)
promo = st.sidebar.selectbox("Promo Planned?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
school_holiday = st.sidebar.selectbox("School Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.sidebar.button("Generate Forecast", type="primary"):
    payload = {
        "store_id": int(store_id),
        "start_date": str(start_date),
        "horizon_days": int(horizon_days),
        "promo": int(promo),
        "school_holiday": int(school_holiday),
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload)
        if res.status_code == 200:
            data = res.json()
            forecast_df = pd.DataFrame(data["forecasts"])

            col1, col2 = st.columns(2)
            col1.metric("Target Store", f"Store #{data['store_id']}")
            col2.metric("Total Projected Sales", f"${data['total_forecasted_sales']:,.2f}")

            # Interactive Plotly Line Chart
            fig = px.line(
                forecast_df,
                x="date",
                y="predicted_sales",
                markers=True,
                title=f"Predicted Daily Sales — Next {horizon_days} Days",
                labels={"predicted_sales": "Units / Sales ($)", "date": "Date"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Daily Detailed Breakdown")
            st.dataframe(forecast_df, use_container_width=True)
        else:
            st.error(f"API Error: {res.text}")
    except Exception as e:
        st.error(f"Failed to connect to API at {API_URL}. Ensure uvicorn is running. Error: {e}")