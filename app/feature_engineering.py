import numpy as np
import pandas as pd


def prepare_features(
    df: pd.DataFrame, store_df: pd.DataFrame = None, is_train: bool = True
) -> pd.DataFrame:
    """Генерация календарных признаков, лагов и скользящих статистик."""
    df = df.copy()

    # 1. Приведение типов и сортировка по времени для каждого магазина
    df['Date'] = pd.to_datetime(df['Date'])
    df['Store'] = df['Store'].astype(int)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # 2. Объединение с метаданными магазинов (store.csv)
    if store_df is not None:
        store_df_copy = store_df.copy()
        store_df_copy['Store'] = store_df_copy['Store'].astype(int)
        df = df.merge(store_df_copy, on='Store', how='left')

    # 3. Календарные признаки
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Приведение категориальных столбцов к типу category для LightGBM
    for col in ['StoreType', 'Assortment', 'StateHoliday']:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category')

    if 'CompetitionDistance' in df.columns:
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
            df['CompetitionDistance'].median()
        )

    # 4. Расчет лагов и скользящих статистик (без data leakage)
    if is_train and 'Sales' in df.columns:
        df['Sales'] = df['Sales'].astype(float)

        # Создаем лаговые колонки
        for lag in [14, 21, 28, 30]:
            df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

        # Вычисляем скользящие средние по созданным лагам
        df['rolling_mean_14'] = df.groupby('Store')['sales_lag_14'].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        df['rolling_std_14'] = df.groupby('Store')['sales_lag_14'].transform(
            lambda x: x.rolling(14, min_periods=1).std()
        )
        df['rolling_mean_30'] = df.groupby('Store')['sales_lag_14'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )

    return df