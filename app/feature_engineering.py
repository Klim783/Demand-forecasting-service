import numpy as np
import pandas as pd


def prepare_features(
    df: pd.DataFrame, store_df: pd.DataFrame = None, is_train: bool = True
) -> pd.DataFrame:
  df = df.copy()

  df['Date'] = pd.to_datetime(df['Date'])
  df['Store'] = df['Store'].astype(int)
  df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

  if store_df is not None:
    store_df_copy = store_df.copy()
    store_df_copy['Store'] = store_df_copy['Store'].astype(int)
    df = df.merge(store_df_copy, on='Store', how='left')

  # 1. Calendar & Time Features
  df['Year'] = df['Date'].dt.year
  df['Month'] = df['Date'].dt.month
  df['Day'] = df['Date'].dt.day
  df['DayOfWeek'] = df['Date'].dt.dayofweek
  df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
  df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

  # 2. Advanced Features: Payday & Cyclical Seasonality
  df['IsPayday'] = df['Day'].isin([1, 2, 15, 16, 30, 31]).astype(int)
  df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
  df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365.25)

  # Categorical Types
  for col in ['StoreType', 'Assortment', 'StateHoliday']:
    if col in df.columns:
      df[col] = df[col].astype(str).astype('category')

  if 'CompetitionDistance' in df.columns:
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
        df['CompetitionDistance'].median()
    )

  # 3. Lags & Target Aggregations
  if 'Sales' in df.columns and is_train:
    open_sales = df[df['Open'] == 1]
    store_day_mean = (
        open_sales.groupby(['Store', 'DayOfWeek'])['Sales']
        .mean()
        .reset_index()
    )
    store_day_mean.rename(
        columns={'Sales': 'store_day_avg_sales'}, inplace=True
    )
    df = df.merge(store_day_mean, on=['Store', 'DayOfWeek'], how='left')

    # Compute lags on open days only
    df['sales_open_only'] = np.where(df['Open'] == 1, df['Sales'], np.nan)
    df['sales_open_only'] = df.groupby('Store')['sales_open_only'].transform(
        lambda x: x.ffill()
    )

    for lag in [14, 21, 28, 30]:
      df[f'sales_lag_{lag}'] = df.groupby('Store')['sales_open_only'].shift(
          lag
      )

    df['rolling_mean_14'] = df.groupby('Store')['sales_lag_14'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df['rolling_std_14'] = df.groupby('Store')['sales_lag_14'].transform(
        lambda x: x.rolling(14, min_periods=1).std()
    )
    df['rolling_mean_30'] = df.groupby('Store')['sales_lag_14'].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    df.drop(columns=['sales_open_only'], inplace=True, errors='ignore')

  return df