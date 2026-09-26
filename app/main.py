import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import StorePredictionRequest, ForecastResponse, DailyForecast
from app.model_loader import ModelLoader
from app.feature_engineering import prepare_features

@asynccontextmanager
async def lifespan(app: FastAPI):
	try:
		ModelLoader.get_model()
		print("Model loaded")
	except Exception as e:
		print(f"Warning on startup: {e}")
	yield

app = FastAPI(
	title="Demand Forecast",
	description="Demand forecast model",
	version="1.0",
	lifespan=lifespan
)
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists("models/lgbm_demand_model.pkl")}


@app.post("/predict", response_model=ForecastResponse)
def predict_demand(payload: StorePredictionRequest):
    try:
        artifact = ModelLoader.get_model()
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate dates horizon
    forecast_dates = pd.date_range(start=payload.start_date, periods=payload.horizon_days, freq="D")

    # Load historical store baseline if available, or generate context
    train_path = "data/train.csv"
    if os.path.exists(train_path):
        history_df = pd.read_csv(train_path, low_memory=False)
        history_df = history_df[history_df["Store"] == payload.store_id].sort_values("Date").tail(60)
    else:
        history_df = pd.DataFrame()

    # Build future horizon dataframe
    future_records = []
    for d in forecast_dates:
        future_records.append({
            "Store": payload.store_id,
            "Date": d.strftime("%Y-%m-%d"),
            "Sales": np.nan,
            "Open": 0 if d.dayofweek == 6 else 1,
            "Promo": payload.promo,
            "StateHoliday": "0",
            "SchoolHoliday": payload.school_holiday,
        })
    future_df = pd.DataFrame(future_records)

    # Combine historical context and future records for lag calculations
    combined = pd.concat([history_df, future_df], ignore_index=True)

    store_path = "data/store.csv"
    store_df = pd.read_csv(store_path) if os.path.exists(store_path) else None

    processed = prepare_features(combined, store_df=store_df, is_train=True)
    future_features = processed.tail(payload.horizon_days).copy()

    # Fill lag NAs if historical context was short
    for col in feature_cols:
        if col in future_features.columns:
            future_features[col] = future_features[col].fillna(future_features[col].median() if not future_features[col].isna().all() else 0)

    X_pred = future_features[feature_cols]
    preds = np.clip(model.predict(X_pred), 0, None)

    daily_forecasts = []
    for i, d in enumerate(forecast_dates):
        # Sunday closed = 0 sales
        val = float(preds[i]) if d.dayofweek != 6 else 0.0
        daily_forecasts.append(DailyForecast(date=d.strftime("%Y-%m-%d"), predicted_sales=round(val, 2)))

    total_sales = sum(f.predicted_sales for f in daily_forecasts)

    return ForecastResponse(
        store_id=payload.store_id,
        horizon_days=payload.horizon_days,
        total_forecasted_sales=round(total_sales, 2),
        forecasts=daily_forecasts,
    )