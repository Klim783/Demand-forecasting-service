import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.feature_engineering import prepare_features


def train_model():
    train_path = 'data/train.csv'
    store_path = 'data/store.csv'

    print(f'Loading Rossmann dataset from {train_path}...')
    df_raw = pd.read_csv(train_path, low_memory=False)

    store_df = None
    if os.path.exists(store_path):
        print(f'Loading store metadata from {store_path}...')
        store_df = pd.read_csv(store_path)

    # Filter out closed stores and zero sales for training
    df_raw = df_raw[(df_raw['Open'] != 0) & (df_raw['Sales'] > 0)].copy()

    # 1. Feature Engineering
    print('Generating time-series features (lags, rolling stats, calendar)...')
    df_processed = prepare_features(
        df_raw, store_df=store_df, is_train=True
    ).dropna()

    # 2. Time-Series Validation Split (last 30 days)
    max_date = df_processed['Date'].max()
    split_date = max_date - pd.Timedelta(days=30)

    train_df = df_processed[df_processed['Date'] <= split_date]
    val_df = df_processed[df_processed['Date'] > split_date]

    feature_cols = [
        'Store',
        'DayOfWeek',
        'Promo',
        'SchoolHoliday',
        'Year',
        'Month',
        'Day',
        'IsWeekend',
        'WeekOfYear',
        'sales_lag_14',
        'sales_lag_21',
        'sales_lag_28',
        'sales_lag_30',
        'rolling_mean_14',
        'rolling_std_14',
        'rolling_mean_30',
    ]

    if 'CompetitionDistance' in df_processed.columns:
        feature_cols.append('CompetitionDistance')

    target_col = 'Sales'

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    # 3. Model Training
    print(
        f'Training LightGBM Regressor on {len(X_train):,} historical rows...'
    )
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    # 4. Evaluation
    val_preds = np.clip(model.predict(X_val), 0, None)

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    wape = np.sum(np.abs(y_val - val_preds)) / np.sum(y_val) * 100

    print('--- Validation Results ---')
    print(f' MAE:  {mae:.2f}')
    print(f' RMSE: {rmse:.2f}')
    print(f' WAPE: {wape:.2f}%')

    # 5. Save Model Artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(
        {'model': model, 'feature_cols': feature_cols},
        'models/lgbm_demand_model.pkl',
    )
    print(' Model successfully saved to models/lgbm_demand_model.pkl')


if __name__ == '__main__':
    train_model()