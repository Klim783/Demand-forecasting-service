import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown(
    "Predict future store sales using **LightGBM Ensemble** with lag features"
    " and promo scenario simulation."
)

# Sidebar Controls
st.sidebar.header("Forecast Parameters")
store_id = st.sidebar.number_input(
    "Store ID", min_value=1, max_value=1115, value=1
)
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2015-08-01")
)
horizon_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=7, max_value=30, value=14
)
promo = st.sidebar.selectbox(
    "Promo Planned?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No",
)
school_holiday = st.sidebar.selectbox(
    "School Holiday?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

st.sidebar.subheader("Historical Baselines (Lags)")
store_day_avg = st.sidebar.number_input(
    "Store Day Avg Sales (€)", value=5200.0, step=100.0
)
lag_14 = st.sidebar.number_input(
    "Sales Lag (14 Days Prior) (€)", value=5100.0, step=100.0
)
rolling_mean_14 = st.sidebar.number_input(
    "14-Day Rolling Mean (€)", value=5050.0, step=100.0
)

if st.sidebar.button("Generate Forecast", type="primary"):
  # 1. Generate full batch payload for each day in the forecast horizon
  payload_list = []
  date_range = pd.date_range(start=start_date, periods=horizon_days)

  for single_date in date_range:
    day_of_week = single_date.weekday()
    day_of_year = single_date.dayofyear
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Sundays are closed by default in Rossmann dataset
    is_open = 0 if day_of_week == 6 else 1

    is_payday = 1 if single_date.day in [1, 2, 15, 16, 30, 31] else 0
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)

    payload_list.append({
        "Store": int(store_id),
        "Open": int(is_open),
        "DayOfWeek": int(day_of_week),
        "Promo": int(promo),
        "SchoolHoliday": int(school_holiday),
        "Year": int(single_date.year),
        "Month": int(single_date.month),
        "Day": int(single_date.day),
        "IsWeekend": int(is_weekend),
        "WeekOfYear": int(single_date.isocalendar()[1]),
        "IsPayday": int(is_payday),
        "Sin_DayOfYear": float(sin_day),
        "Cos_DayOfYear": float(cos_day),
        "sales_lag_14": float(lag_14),
        "sales_lag_21": float(lag_14 * 0.98),
        "sales_lag_28": float(lag_14 * 0.95),
        "sales_lag_30": float(lag_14 * 0.96),
        "rolling_mean_14": float(rolling_mean_14),
        "rolling_std_14": 300.0,
        "rolling_mean_30": float(rolling_mean_14 * 0.97),
        "store_day_avg_sales": float(store_day_avg),
    })

  try:
    # Send vectorized batch request to FastAPI
    res = requests.post(f"{API_URL}/predict-batch", json=payload_list)

    if res.status_code == 200:
      data = res.json()

      # Construct results DataFrame
      predictions = [
          item["predicted_sales_eur"] for item in data["batch_predictions"]
      ]
      forecast_df = pd.DataFrame({
          "date": date_range.strftime("%Y-%m-%d"),
          "predicted_sales": predictions,
      })

      total_sales = forecast_df["predicted_sales"].sum()

      # Metrics Cards
      col1, col2 = st.columns(2)
      col1.metric("Target Store", f"Store #{store_id}")
      col2.metric("Total Projected Sales", f"€{total_sales:,.2f}")

      # Plotly Express Line Chart
      fig = px.line(
          forecast_df,
          x="date",
          y="predicted_sales",
          markers=True,
          title=f"Predicted Daily Sales — Next {horizon_days} Days",
          labels={"predicted_sales": "Predicted Sales (€)", "date": "Date"},
      )
      fig.update_traces(
          line_color="#2ca02c",
          line_width=3,
          marker=dict(size=8, symbol="circle"),
      )
      fig.update_layout(hovermode="x unified")

      st.plotly_chart(fig, use_container_width=True)

      # Detailed Dataframe Breakdown
      st.subheader("Daily Detailed Breakdown")
      st.dataframe(forecast_df, use_container_width=True)

    else:
      st.error(f"API Error: {res.text}")

  except Exception as e:
    st.error(
        f"Failed to connect to API at {API_URL}. Ensure uvicorn is running."
        f" Error: {e}"
    )