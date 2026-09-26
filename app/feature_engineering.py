import numpy as np
import pandas as pd

def prepare_features(df:pd.DataFrame, store_df: pd.DataFrame = None, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df =  df.sort_values(["Store", "Date"]).reset_index(drop = True)

    if store_df is not None:
        df = df.merge(store_df, on = "Store", how = "left")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    for col in ["StoreType", "Assortment"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    if "Sales" in df.columns:
        for lag in [14,21,28,30]:
            df["sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
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