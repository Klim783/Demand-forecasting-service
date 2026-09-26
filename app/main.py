from typing import List
import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Rossmann Store Sales Demand Predictor API", version="2.0.0"
)

# Load machine learning artifacts
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
  Store: int = Field(..., alias="store_id")
  Open: int = 1
  DayOfWeek: int = 4
  Promo: int = 1
  SchoolHoliday: int = 0
  Year: int = 2015
  Month: int = 7
  Day: int = 31
  IsWeekend: int = 0
  WeekOfYear: int = 31
  IsPayday: int = 1
  Sin_DayOfYear: float = -0.493
  Cos_DayOfYear: float = -0.870
  sales_lag_14: float = 5000.0
  sales_lag_21: float = 5000.0
  sales_lag_28: float = 5000.0
  sales_lag_30: float = 5000.0
  rolling_mean_14: float = 5000.0
  rolling_std_14: float = 300.0
  rolling_mean_30: float = 5000.0
  store_day_avg_sales: float = 5000.0

  class Config:
    populate_by_name = True


def extract_dict(pydantic_obj: BaseModel) -> dict:
  """Helper for cross-version Pydantic dictionary dump."""
  if hasattr(pydantic_obj, "model_dump"):
    return pydantic_obj.model_dump(by_alias=False)
  return pydantic_obj.dict(by_alias=False)


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
    data_dict = extract_dict(payload)
    input_df = pd.DataFrame([data_dict])[feature_cols]

    # Ensemble Averaging Across All Folds
    preds_log = [model.predict(input_df)[0] for model in models]
    avg_pred_log = float(np.mean(preds_log))
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
    # Safely dump Pydantic objects to dicts
    batch_dicts = [extract_dict(p) for p in payload_list]
    df_batch = pd.DataFrame(batch_dicts)

    # Ensure all required features exist in the DataFrame
    missing_cols = [c for c in feature_cols if c not in df_batch.columns]
    if missing_cols:
      raise ValueError(f"Missing required features: {missing_cols}")

    # Ensemble batch predictions across folds
    preds_log_matrix = np.column_stack(
        [model.predict(df_batch[feature_cols]) for model in models]
    )
    avg_preds_log = np.mean(preds_log_matrix, axis=1)
    predicted_sales = np.expm1(avg_preds_log)

    # Post-processing override for closed stores
    predicted_sales = np.where(df_batch["Open"] == 0, 0.0, predicted_sales)

    # Access fields directly via dot notation on the Pydantic object
    results = [
        {
            "store_id": item.Store,
            "predicted_sales_eur": round(max(0.0, float(pred)), 2),
        }
        for item, pred in zip(payload_list, predicted_sales)
    ]

    return {"batch_predictions": results}
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))