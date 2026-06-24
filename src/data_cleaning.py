import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt
import src.saving_output as so

def check_holiday_dates(df, holiday_dates):
    """
    Check consistency of holiday flags against a reference list of holiday dates.
    """
    # Compare dataset dates with reference holiday list
    df['IsHoliday_list'] = df['Date'].isin(holiday_dates)
    # Identify inconsistencies between different holiday indicators
    df['Holiday_mismatch'] = ~(
    (df['IsHoliday_x'] == df['IsHoliday_y']) &
    (df['IsHoliday_x'] == df['IsHoliday_list']))
    # Extract inconsistent observations
    errors = df.loc[df['Holiday_mismatch'], ['Date', 'IsHoliday_x', 'IsHoliday_y', 'IsHoliday_list']]
    print(f"Number of inconsistencies: {errors.shape[0]}")
    print(f"\n Inconsistent dates: \n{errors}")
    return errors

def n_missing_values(df):
    """
    Print number of missing values per column.
    """
    print(f'Number of missing values: \n\n{df.isna().sum()}')

def missing_values_imputation(df, cols, method=None, value=None):
    """
    Fill missing values in selected columns using a chosen method or a fixed value.
    """
    for col in cols:
        if method == 'ffill':
            df[col] = df[col].ffill()
        elif method == 'bfill':
            df[col] = df[col].bfill()
        elif value is not None:
            df[col] = df[col].fillna(value=value)
        else:
            raise ValueError("You must provide either a method or a value for imputation.")
    return df


def weekly_sales_distribution_boxplot(train_full_20_1, folders):
    """
    Plot distribution of weekly sales using a boxplot.
    """
    plt.figure()
    plt.boxplot(train_full_20_1['Weekly_Sales'])
    plt.title('Weekly Sales Distribution', fontsize=16)
    median_value_weekly_sales_train = train_full_20_1['Weekly_Sales'].median()
    plt.axhline(y=median_value_weekly_sales_train, alpha=0.5, linestyle='--',
                label=f'median = {median_value_weekly_sales_train: .2f}', color='orange')
    plt.legend()
    plt.grid(True)
    so.save_plot("weekly_sales_distribution_boxplot", folders)
    # plt.show()


def weekly_sales_distribution_by_holiday(train_full_20_1, folders):
    """
    Plot weekly sales distribution grouped by holiday vs non-holiday periods.
    """
    plt.figure()
    train_full_20_1.boxplot(column='Weekly_Sales', by='IsHoliday')
    plt.title('Weekly Sales Distribution by Holiday', fontsize=16)
    plt.suptitle('')  # rimuove titolo automatico
    plt.xlabel('IsHoliday', fontsize=14)
    plt.ylabel('Weekly Sales', fontsize=14)
    median_value_weekly_sales_train = train_full_20_1['Weekly_Sales'].median()
    plt.axhline(y=median_value_weekly_sales_train, alpha=0.5, linestyle='--',
                label=f'median = {median_value_weekly_sales_train: .2f}', color='green')
    plt.grid(True)
    plt.legend()
    so.save_plot("weekly_sales_distribution_by_holiday", folders)
    # plt.show()


def weekly_sales_distribution_by_holiday_period(train_full_20_1, folders):
    """
    Plot weekly sales distribution comparing holiday and non-holiday periods.
    """
    plt.figure()
    train_full_20_1.boxplot(column='Weekly_Sales', by='IsHolidayPeriod')
    plt.title('Weekly Sales Distribution by Holiday Period', fontsize=16)
    plt.suptitle('')
    plt.xlabel('Holiday Period', fontsize=14)
    plt.ylabel('Weekly Sales', fontsize=14)

    plt.xticks(
        ticks=[1, 2],
        labels=['Non-holiday period', 'Holiday period'], fontsize=14)

    so.save_plot("weekly_sales_distribution_by_holiday_period", folders)
    # plt.show()


def feature_averages_comparison_overall_vs_sales_outliers(train_full_20_1, folders):
    """
    Compare feature averages between all data and high sales outliers.
    """
    q1 = train_full_20_1['Weekly_Sales'].quantile(0.25)
    q3 = train_full_20_1['Weekly_Sales'].quantile(0.75)
    iqr = q3 - q1

    outlier_threshold = q3 + 1.5 * iqr

    train_full_20_1['IsOutlier'] = (
            train_full_20_1['Weekly_Sales'] > outlier_threshold
    ).astype(int)

    features_to_analyze = [
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5'
    ]

    overall_means = train_full_20_1[features_to_analyze].mean()
    outlier_means = (train_full_20_1[train_full_20_1['IsOutlier'] == 1]
                     [features_to_analyze]
                     .mean()
                     )
    mean_comparison = pd.DataFrame({
        'Overall mean': overall_means,
        'Outlier mean': outlier_means
    })

    ax = mean_comparison.plot(
        kind='bar',
        figsize=(20, 10)
    )

    ax.set_title('Feature Averages: Overall vs Sales Outliers', fontsize=26)
    ax.set_ylabel('Average value', fontsize=20)
    ax.set_xlabel('Feature', fontsize=20)
    ax.legend(fontsize=20)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3,label_type='center', fontsize=11)

    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    so.save_plot("feature_averages_comparison_overall_vs_sales_outliers", folders)
    # plt.show()