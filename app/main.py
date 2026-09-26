from typing import List
import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Rossmann Store Sales Demand Predictor API", version="2.0.0"
)

try:
  artifact = joblib.load("models/lgbm_demand_model.pkl")
  models = artifact.get(
      "models", [artifact.get("model")]
  )  # Supports single model or ensemble
  feature_cols = artifact["feature_cols"]
  metrics = artifact.get("metrics", {})
except Exception as e:
  models = []
  feature_cols = []
  metrics = {}
  print(f"Error loading model: {e}")


class StoreFeaturesInput(BaseModel):
  Store: int
  Open: int = Field(1, description="1 = Open, 0 = Closed")
  DayOfWeek: int
  Promo: int
  SchoolHoliday: int
  Year: int
  Month: int
  Day: int
  IsWeekend: int
  WeekOfYear: int
  IsPayday: int
  Sin_DayOfYear: float
  Cos_DayOfYear: float
  sales_lag_14: float
  sales_lag_21: float
  sales_lag_28: float
  sales_lag_30: float
  rolling_mean_14: float
  rolling_std_14: float
  rolling_mean_30: float
  store_day_avg_sales: float


@app.get("/")
def health_check():
  return {
      "status": "ok" if len(models) > 0 else "model_not_loaded",
      "ensemble_size": len(models),
      "validation_metrics": metrics,
  }


@app.post("/predict")
def predict_sales(payload: StoreFeaturesInput):
  if not models:
    raise HTTPException(status_code=500, detail="Model artifact missing.")

  # Business Rule: Closed stores have zero sales
  if payload.Open == 0:
    return {"store_id": payload.Store, "predicted_sales_eur": 0.0}

  try:
    data_dict = payload.model_dump()
    input_df = pd.DataFrame([data_dict])[feature_cols]

    # Ensemble Averaging Across All Folds
    preds_log = [model.predict(input_df)[0] for model in models]
    avg_pred_log = np.mean(preds_log)
    predicted_sales = float(np.expm1(avg_pred_log))

    return {
        "store_id": payload.Store,
        "predicted_sales_eur": round(max(0.0, predicted_sales), 2),
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
def predict_batch_sales(payload_list: List[StoreFeaturesInput]):
  if not models:
    raise HTTPException(status_code=500, detail="Model artifact missing.")

  try:
    df_batch = pd.DataFrame([p.model_dump() for p in payload_list])

    # Ensemble batch predictions
    preds_log_list = np.column_stack(
        [model.predict(df_batch[feature_cols]) for model in models]
    )
    avg_preds_log = np.mean(preds_log_list, axis=1)
    predicted_sales = np.expm1(avg_preds_log)

    # Post-processing override for closed stores
    predicted_sales = np.where(df_batch["Open"] == 0, 0.0, predicted_sales)

    results = [
        {"store_id": row["Store"], "predicted_sales_eur": round(max(0.0, p), 2)}
        for row, p in zip(payload_list, predicted_sales)
    ]
    return {"batch_predictions": results}
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))