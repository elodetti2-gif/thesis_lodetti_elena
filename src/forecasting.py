import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp
import src.saving_output as so
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt

def final_forecasting_datasets(train_full_20_1, test_full_20_1):
    """
    Prepare final training and test datasets for forecasting.
    """
    X_train_final = train_full_20_1.drop(columns=["Weekly_Sales", "Date", 'IsOutlier'])
    y_train_final = train_full_20_1["Weekly_Sales"]
    
    X_test = test_full_20_1.drop(columns=["Weekly_Sales","Date", 'IsOutlier'])
    return X_train_final, y_train_final, X_test

def final_model_rf_definition():
    """
    Define final Random Forest model for forecasting.
    """
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    return final_model
    
def final_model_forecasting(X_train_final, y_train_final, final_model, test_full_20_1, X_test):
    """
    Train final model and generate predictions on test dataset.
    """
    final_model.fit(X_train_final, y_train_final)
    test_full_20_1["Predicted_Weekly_Sales"] = final_model.predict(X_test)
    return test_full_20_1


def forecasted_weekly_sales_plot(test_full_20_1, folders):
    """
    Plot forecasted weekly sales over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_full_20_1["Date"], test_full_20_1["Predicted_Weekly_Sales"], label='Predicted sales over time')
    ax.set_title("Forecasted Weekly Sales on Test Set", fontsize=18)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Weekly Sales", fontsize=16)

    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=14)
    so.save_plot("forcasted_weekly_sales_plot", folders)
    # plt.show()

