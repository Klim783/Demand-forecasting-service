import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from app.feature_engineering import prepare_features


def rmpspe_metric(y_true, y_pred):
  mask = y_true > 0
  return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100


def train_oof_pipeline():
  print("--- Starting Production Out-of-Fold Ensemble Training ---")

  df_raw = pd.read_csv("data/train.csv", low_memory=False)
  store_df = pd.read_csv("data/store.csv")

  # Filter out closed days & abnormal outliers
  df_raw = df_raw[(df_raw["Open"] == 1) & (df_raw["Sales"] > 0)].copy()

  df_processed = prepare_features(df_raw, store_df=store_df, is_train=True).dropna()

  feature_cols = [
      "Store",
      "DayOfWeek",
      "Promo",
      "SchoolHoliday",
      "Year",
      "Month",
      "Day",
      "IsWeekend",
      "WeekOfYear",
      "IsPayday",
      "Sin_DayOfYear",
      "Cos_DayOfYear",
      "sales_lag_14",
      "sales_lag_21",
      "sales_lag_28",
      "sales_lag_30",
      "rolling_mean_14",
      "rolling_std_14",
      "rolling_mean_30",
      "store_day_avg_sales",
  ]

  for col in feature_cols:
    if col in df_processed.columns:
      df_processed[col] = df_processed[col].fillna(df_processed[col].median())

  X = df_processed[feature_cols]
  y = df_processed["Sales"].values
  y_log = np.log1p(y)

  # 5-Fold TimeSeriesSplit Cross Validation
  tscv = TimeSeriesSplit(n_splits=5)
  models = []
  oof_preds = np.zeros(len(X))

  print("Training 5 LightGBM Fold Models...")
  for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, y_train_log = X.iloc[train_idx], y_log[train_idx]
    X_val, y_val_log = X.iloc[val_idx], y_log[val_idx]

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train_log)
    val_pred_log = model.predict(X_val)
    oof_preds[val_idx] = np.expm1(val_pred_log)
    models.append(model)
    print(f" Fold {fold + 1} Trained Successfully.")

  # Evaluate Out-of-Fold Performance on valid splits
  eval_idx = np.where(oof_preds > 0)[0]
  mae = mean_absolute_error(y[eval_idx], oof_preds[eval_idx])
  rmse = np.sqrt(mean_squared_error(y[eval_idx], oof_preds[eval_idx]))
  rmspe = rmpspe_metric(y[eval_idx], oof_preds[eval_idx])

  print("\n" + "=" * 45)
  print("      5-FOLD OUT-OF-FOLD METRICS      ")
  print("=" * 45)
  print(f" OOF MAE:   €{mae:.2f}")
  print(f" OOF RMSE:  €{rmse:.2f}")
  print(f" OOF RMSPE: {rmspe:.2f}%")
  print("=" * 45 + "\n")

  # Save Ensembled Models & Artifacts
  os.makedirs("models", exist_ok=True)
  artifact = {
      "models": models,
      "feature_cols": feature_cols,
      "metrics": {"mae": mae, "rmse": rmse, "rmspe": rmspe},
  }
  joblib.dump(artifact, "models/lgbm_demand_model.pkl")
  print("OOF Ensemble saved to 'models/lgbm_demand_model.pkl'!")


if __name__ == "__main__":
  train_oof_pipeline()