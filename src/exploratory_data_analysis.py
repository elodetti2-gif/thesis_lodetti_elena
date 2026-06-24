import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt
import shap
import src.saving_output as so

#I create a graph that shows the weekly sales in function of the date in the train set
def weekly_sales_x_week(train_20_1, folders):
    """
    Plot weekly sales over time and highlight high values.
    """
    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(train_20_1['Date'], train_20_1['Weekly_Sales'])
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Weekly Sales', fontsize=14)
    ax.set_title('Weekly Sales - Train set', fontsize=16)

    threshold=train_20_1['Weekly_Sales'].mean()+train_20_1['Weekly_Sales'].std()
    mask = train_20_1['Weekly_Sales'] >= threshold

    ax.scatter(train_20_1.loc[mask, 'Date'], train_20_1.loc[mask, 'Weekly_Sales'],color='red', s=50)
    ax.set_xticks(train_20_1['Date'][::4])
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    high_weeks_unique = train_20_1.loc[train_20_1['Weekly_Sales'] >= threshold,'Date'].unique()

    for tick in ax.get_xticklabels():
        if tick.get_text() in high_weeks_unique:
            tick.set_color('red')
            tick.set_fontweight('bold')

    #ax.legend()
    plt.grid(True)
    plt.tight_layout()
    so.save_plot('weekly_sales_x_week_train_20_1', folders)
    #plt.show()
    
def plotting_basic(df, column, folders):
    """
    Plot relationship between a feature and weekly sales using scatter plot.
    """
    plt.figure(figsize=(7.7,5.7))
    plt.scatter(df[column], df['Weekly_Sales'], alpha=0.5, s=25)
    plt.xlabel(column, fontsize=20)
    plt.ylabel('Weekly Sales', fontsize=20)
    plt.title(f'Weekly Sales vs {column}', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    fig_name=f"scatter_weekly_sales_{column}"
    so.save_plot(fig_name, folders)
    #plt.show()
    
def avg_weekly_sales_by_week_type(df, folders):
    """
    Plot average weekly sales comparing holiday and non-holiday weeks.
    """
    df.groupby('IsHoliday')['Weekly_Sales'].mean().plot(kind='bar')
    plt.ylabel('Average Weekly Sales', fontsize=16)
    plt.title('Average Weekly Sales by Week Type', fontsize=18)
    plt.yticks(df.groupby('IsHoliday')['Weekly_Sales'].mean(), fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(True, axis='y')
    so.save_plot("avg_weekly_sales_by_week_type",folders)
    #plt.show()

def avg_weekly_sales_by_time_period(df, period, folders):
    """
    Plot average weekly sales aggregated by a time period (year, month, or week).
    """
    period_avg = df.groupby(period)['Weekly_Sales'].mean().sort_index()
    fig, ax = plt.subplots()
    bars = ax.bar(period_avg.index, period_avg.values)
    ax.set_title(f'Average Weekly Sales by {period}', fontsize=18)
    ax.set_xlabel(f'{period}', fontsize=16)
    ax.set_ylabel('Average Weekly Sales', fontsize=16)
    #ax.bar_label(bars, fmt='%.0f', padding=3, label_type='center', fontsize=16)
    plt.grid(True, axis='y')
    fig_name=f"avg_weekly_sales_by_{period}"
    so.save_plot(fig_name, folders)
    #plt.show()

def avg_weekly_sales_by_week_of_the_year(train_full_20_1):
    weekly_avg = train_full_20_1.groupby('Week')['Weekly_Sales'].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(weekly_avg.index, weekly_avg.values)

    ax.set_title('Average Weekly Sales by Week of the Year', fontsize=18)
    ax.set_xlabel('Week', fontsize=16)
    ax.set_ylabel('Average Weekly Sales', fontsize=16)
    ax.set_xticks(weekly_avg.index, fontsize=14)
    ax.tick_params(axis='x', rotation=-90)

    threshold=weekly_avg.mean() + weekly_avg.std()

    highlight_weeks=[]

    for week in weekly_avg.index:
        if weekly_avg[week]>=threshold:
            ax.scatter(week, weekly_avg[week], color='red', marker='o')
            ax.text(week+1,weekly_avg[week]+1, f'{weekly_avg[week]:.0f}', fontsize=12)
    highlight_weeks.append(week)

    for tick in ax.get_xticklabels():
        if int(tick.get_text()) in highlight_weeks:
            tick.set_color('red')
            tick.set_fontweight('bold')

    plt.grid(True, alpha=0.5)
    #plt.show()
    




        


