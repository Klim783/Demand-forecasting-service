from datetime import datetime, date
from typing import List, Optional
from pydantic import BaseModel, Field

class StorePredictionRequest(BaseModel):
    store_id: int = Field(..., example=1, description="Unique Store ID")
    start_date: date = Field(..., example="2015-08-01", description="Prediction start date")
    horizon_days: int = Field(default=14, ge=1, le=42, description="Forecast horizon in days")
    promo: int = Field(default=0, ge=0, le=1, description="Planned promo active (0 or 1)")
    school_holiday: int = Field(default=0, ge=0, le=1, description="School holiday indicator (0 or 1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "store_id": 1,
                "start_date": "2015-08-01",
                "horizon_days": 14,
                "promo": 1,
                "school_holiday": 0,
            }
        }
    }

class DailyForecast(BaseModel):
    date:str
    predicted_sales: float

class ForecastResponse(BaseModel):
    store_id:int
    horizon_days: int
    total_forecasted_sales: float
    forecasts: List[DailyForecast]
