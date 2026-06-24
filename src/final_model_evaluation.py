import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt
import shap
import src.saving_output as so
from scipy.stats import ks_2samp

def historical_predicted_weekly_sales_distributions_box_plot(y_train_final, test_full_20_1, folders):
    """
    Compare distribution of historical and predicted weekly sales using boxplots.
    """
    fig, axs = plt.subplots(1, 2, figsize=(6,10))

    def box_stats(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]

        q1 = np.percentile(x, 25)
        q2 = np.percentile(x, 50)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        lower_whisker = x[x >= lower_fence].min()
        upper_whisker = x[x <= upper_fence].max()

        return q1, q2, q3, iqr, lower_whisker, upper_whisker

    # Historical Weekly Sales
    x_train = y_train_final
    axs[0].boxplot(x_train, medianprops={'color': 'red'})
    axs[0].set_ylabel("Weekly Sales", fontsize=16)
    axs[0].set_title("Historical Weekly Sales", color='red', fontsize=18)
    axs[0].set_xticks([])

    Q1_real_train, Q2_real_train, Q3_real_train, IQR_real_train, lower_whisker_real_train, upper_whisker_real_train = box_stats(
        x_train)

    axs[0].axhline(Q1_real_train, color="red", alpha=0.15, label=f"25° Percentile: {Q1_real_train:.0f}")
    axs[0].axhline(Q3_real_train, color="red", alpha=0.15, label=f"75° Percentile: {Q3_real_train:.0f}")
    axs[0].axhline(Q2_real_train, color="red", alpha=0.25, label=f"Median: {Q2_real_train:.0f}")
    axs[0].axhline(upper_whisker_real_train, color="red", alpha=0.15,
                   label=f"Upper Whisker: {upper_whisker_real_train:.0f}")
    axs[0].axhline(lower_whisker_real_train, color="red", alpha=0.15,
                   label=f"Lower Whisker: {lower_whisker_real_train:.0f}")

    axs[0].set_yticks(sorted(set([
        Q1_real_train, Q2_real_train, Q3_real_train,
        lower_whisker_real_train, upper_whisker_real_train
    ])))

    # Predicted Weekly Sales
    x_pred = test_full_20_1["Predicted_Weekly_Sales"]
    axs[1].boxplot(x_pred, medianprops={'color': 'blue'})
    axs[1].set_title("Predicted Weekly Sales", color='blue', fontsize=18)
    axs[1].set_xticks([])

    Q1_predicted_test, Q2_predicted_test, Q3_predicted_test, IQR_predicted_test, lower_whisker_predicted_test, upper_whisker_predicted_test = box_stats(
        x_pred)

    axs[1].axhline(Q1_predicted_test, color="blue", alpha=0.15, label=f"25° Percentile: {Q1_predicted_test:.0f}")
    axs[1].axhline(Q3_predicted_test, color="blue", alpha=0.15, label=f"75° Percentile: {Q3_predicted_test:.0f}")
    axs[1].axhline(Q2_predicted_test, color="blue", alpha=0.25, label=f"Median: {Q2_predicted_test:.0f}")
    axs[1].axhline(upper_whisker_predicted_test, color="blue", alpha=0.15,
                   label=f"Upper Whisker: {upper_whisker_predicted_test:.0f}")
    axs[1].axhline(lower_whisker_predicted_test, color="blue", alpha=0.15,
                   label=f"Lower Whisker: {lower_whisker_predicted_test:.0f}")

    axs[1].set_yticks(sorted(set([
        Q1_predicted_test, Q2_predicted_test, Q3_predicted_test,
        lower_whisker_predicted_test, upper_whisker_predicted_test
    ])))

    fig.suptitle("Historical and Predicted Weekly Sales Distributions", fontsize=16)
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.25), fontsize=14)

    plt.tight_layout()
    so.save_plot("historical_predicted_weekly_sales_distributions_box_plot", folders)
    # plt.show()


def historical_predicted_weekly_sales_distribution_histogram(y_train_final, test_full_20_1, folders):
    """
    Compare distributions of historical and predicted weekly sales using histograms.
    """
    fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(11, 5))

    axs[0].hist(y_train_final, density=True, bins=40, alpha=0.4, color="red")
    axs[0].set_title("Historical Weekly Sales", color="red", fontsize=16)
    axs[0].set_xlabel("Weekly Sales", fontsize=14)
    axs[0].set_ylabel("Density", fontsize=14)

    axs[1].hist(test_full_20_1["Predicted_Weekly_Sales"], density=True, bins=40, alpha=0.3, color="blue")
    axs[1].set_title("Predicted Weekly Sales", color="blue", fontsize=16)
    axs[1].set_xlabel("Weekly Sales", fontsize=14)
    axs[1].set_ylabel("Density", fontsize=14)

    fig.suptitle("Historical and Predicted Weekly Sales Distributions", fontsize=18)

    axs[0].grid(True, alpha=0.25)
    axs[1].grid(True, alpha=0.25)

    axs[0].axvline(y_train_final.median(), linestyle="--", linewidth=1, color="black",
                   label=f'Historical Weekly Sales median = {y_train_final.median():,.0f}')
    axs[1].axvline(test_full_20_1["Predicted_Weekly_Sales"].median(), linestyle="--", linewidth=1, color="black",
                   label=f"Predicted Weekly Sales median = {test_full_20_1['Predicted_Weekly_Sales'].median():,.0f}")
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)

    plt.tight_layout()
    so.save_plot("histroical_predicted_weekly_sales_distribution_histogram", folders)
    # plt.show()

def kolomogorov_smirnov_test(y_train_final, test_full_20_1, folders):
    """
    Perform Kolmogorov-Smirnov test to compare real and predicted distributions.
    """
    ks_stat, p_value = ks_2samp(
        y_train_final,
        test_full_20_1["Predicted_Weekly_Sales"],
        alternative="two-sided",
        method="auto"
    )

    print(f"KS statistic: {ks_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    text=f"KOLOMOGOROV-SMIRNOV TEST RESULTS:\nKS statistic: {ks_stat}\np-value: {p_value}"
    so.save_text(text, "kolomogorov_smirnov_test", folders)
    return ks_stat, p_value