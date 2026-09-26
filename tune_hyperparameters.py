import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from app.feature_engineering import prepare_features


def tune_lgbm():
  print('Загрузка данных...')
  df_raw = pd.read_csv('data/train.csv', low_memory=False)
  store_df = pd.read_csv('data/store.csv')

  # Фильтрация открытых дней
  df_raw = df_raw[(df_raw['Open'] != 0) & (df_raw['Sales'] > 0)].copy()

  print('Подготовка признаков...')
  df_processed = prepare_features(
      df_raw, store_df=store_df, is_train=True
  ).dropna()

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
      'store_day_avg_sales',
  ]

  X = df_processed[feature_cols]
  y_log = np.log1p(df_processed['Sales'])

  # Пространство гиперпараметров
  param_distributions = {
      'n_estimators': [500, 800, 1200],
      'learning_rate': [0.01, 0.02, 0.05],
      'num_leaves': [31, 63, 127, 255],
      'max_depth': [6, 8, 10, 12],
      'subsample': [0.7, 0.8, 0.9],
      'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
      'min_child_samples': [20, 50, 100],
      'reg_alpha': [0.0, 0.1, 1.0],
      'reg_lambda': [0.0, 1.0, 5.0],
  }

  # TimeSeriesSplit для временных рядов (3 фолда)
  tscv = TimeSeriesSplit(n_splits=3)

  base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

  search = RandomizedSearchCV(
      estimator=base_model,
      param_distributions=param_distributions,
      n_iter=15,  # Количество случайных комбинаций
      scoring='neg_mean_squared_error',
      cv=tscv,
      verbose=2,
      random_state=42,
      n_jobs=-1,
  )

  print('Запуск RandomizedSearchCV...')
  search.fit(X, y_log)

  print('\n=== Лучшие гиперпараметры ===')
  print(search.best_params_)

  # Сохранение лучшей модели
  best_model = search.best_estimator_
  joblib.dump(
      {'model': best_model, 'feature_cols': feature_cols},
      'models/lgbm_demand_model.pkl',
  )
  print('Оптимизированная модель сохранена в models/lgbm_demand_model.pkl')


if __name__ == '__main__':
  tune_lgbm()