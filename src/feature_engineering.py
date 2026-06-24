import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_holiday_period_variable(train_full_20_1, test_full_20_1):
    """
    Create a binary feature indicating holiday periods using a rolling window.
    """
    train_full_20_1 = train_full_20_1.sort_values('Date')
    test_full_20_1 = test_full_20_1.sort_values('Date')

    train_full_20_1['IsHolidayPeriod'] = (
        train_full_20_1['IsHoliday']
        .rolling(window=3, center=True)
        .max()
        .fillna(0)
        .astype(int)
    )
    return train_full_20_1, test_full_20_1

def calendar_features(df):
    """
    Extract calendar-based features from the Date column.
    """
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df
    
def lag_features_roll_mean_features(df1, df2):
    """
    Create lag and rolling mean features for time series forecasting.
    """

    #Ensure that the Date column is in datetime format
    df1["Date"] = pd.to_datetime(df1["Date"])
    df2["Date"]  = pd.to_datetime(df2["Date"])

    #Create a single temporary dataframe combining train and test data
    #Sort the combined dataframe by Date
    tmp = pd.concat([df1, df2],     ignore_index=True).sort_values("Date")

    #Create lag features of Weekly_Sales
    tmp["lag_1"]  = tmp["Weekly_Sales"].shift(1)
    tmp["lag_2"]  = tmp["Weekly_Sales"].shift(2)
    tmp["lag_4"]  = tmp["Weekly_Sales"].shift(4)
    tmp["lag_52"] = tmp["Weekly_Sales"].shift(52)

    #Create rolling mean features of Weekly_Sales
    tmp["roll_mean_4"]  = tmp["Weekly_Sales"].shift(1).rolling(4).mean()
    tmp["roll_mean_12"] = tmp["Weekly_Sales"].shift(1).rolling(12).mean()
    df1 = tmp[tmp["Weekly_Sales"].notna()].copy()
    df2  = tmp[tmp["Weekly_Sales"].isna()].copy()

    #Remove training rows with missing lag or rolling features
    df1 = df1.dropna(subset=["lag_1","lag_4","roll_mean_4"])
    df1 = df1.sort_values("Date")
    df2=df2.sort_values("Date")
    return df1, df2