import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt
import shap
import src.saving_output as so

def residual_analysis(model_values, folders):
    """
    Plot residuals over time for each fold of TimeSeriesSplit.
    """
    n_folds = len(model_values['residuals_per_fold'])

    fig, axes = plt.subplots(n_folds, 1, figsize=(12, 4 * n_folds), sharex=False)

    # caso speciale: se c'è 1 solo fold
    if n_folds == 1:
        axes = [axes]

    for i in range(n_folds):

        residuals = model_values['residuals_per_fold'][i]
        year_week = model_values['year_week_per_fold'][i]

        x = range(len(residuals))
        labels = [f"{y}-W{w:02d}" for y, w in year_week]

        ax = axes[i]

        ax.plot(x, residuals, marker='o', linestyle='-')
        ax.axhline(0, linestyle='--')

        ax.set_title(
            f"Fold {i+1}\n{model_values['model_name']} on TimeSeriesSplit", fontsize=18
        )

        ax.set_ylabel("Residuals", fontsize=16)

        step = max(len(x) // 8, 1)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(labels[::step], rotation=45, fontsize=14)

        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel("Week (Year–Week)", fontsize=16)

    plt.tight_layout()

    fig_name = f"residual_analysis_{model_values['model_name']}_TSS_all_folds"
    so.save_plot(fig_name, folders)

    #plt.show()

def comparison_predicted_actual_weekly_sales_TSS(model_values, folders):
    """
    Compare predicted and actual weekly sales for each fold.
    """
    n_folds = len(model_values['y_val_per_fold'])

    fig, axes = plt.subplots(n_folds, 1, figsize=(12, 4 * n_folds), sharex=False)

    # caso 1 fold
    if n_folds == 1:
        axes = [axes]

    for i in range(n_folds):

        ax = axes[i]

        x_labels = [
            f"{y}-W{w}"
            for (y, w) in model_values['year_week_per_fold'][i]
        ]

        ax.set_title(
            f"Fold {i+1} - Predicted vs Actual\n"
            f"{model_values['model_name']} on TimeSeriesSplit", fontsize=16
        )

        ax.plot(
            x_labels,
            model_values['y_val_per_fold'][i],
            label="Actual"
        )

        ax.plot(
            x_labels,
            model_values['y_pred_per_fold'][i],
            label="Predicted"
        )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=14)

        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel("Week (Year–Week)", fontsize=14)

    plt.tight_layout()

    fig_name = f"cfr_predicted_actual_sales_all_folds_{model_values['model_name']}_TSS"
    so.save_plot(fig_name, folders)

    #plt.show()

def model_comparison_lr_rf(model_values_1, model_values_2, folders):
    """
    Compare predictions of Linear Regression and Random Forest across folds.
    """
    n_folds = len(model_values_1['y_val_per_fold'])

    fig, axes = plt.subplots(n_folds, 1, figsize=(14, 4 * n_folds), sharex=False)

    # caso 1 fold
    if n_folds == 1:
        axes = [axes]

    for i in range(n_folds):

        ax = axes[i]

        # asse x
        x_labels = [
            f"{y}-W{w}"
            for (y, w) in model_values_1['year_week_per_fold'][i]
        ]

        ax.set_title(
            f"Fold {i+1} - Linear Regression vs Random Forest", fontsize=16
        )

        # Actual
        ax.plot(
            x_labels,
            model_values_1['y_val_per_fold'][i],
            label="Actual"
        )

        # Model 1
        ax.plot(
            x_labels,
            model_values_1['y_pred_per_fold'][i],
            label=model_values_1['model_name']
        )

        # Model 2
        ax.plot(
            x_labels,
            model_values_2['y_pred_per_fold'][i],
            label=model_values_2['model_name']
        )

        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=14)

    axes[-1].set_xlabel("Week (Year–Week)", fontsize=14)

    plt.tight_layout()

    fig_name = "model_comparison_lr_rf_all_folds"
    so.save_plot(fig_name, folders)

    #plt.show()


def residual_analysis_comparison_lr_rf(model_values_1, model_values_2, folders):
    """
    Compare residuals of Linear Regression and Random Forest across folds.
    """
    n_folds = len(model_values_1['residuals_per_fold'])

    fig, axes = plt.subplots(n_folds, 1, figsize=(12, 4 * n_folds), sharex=False)

    # se n_folds = 1, axes non è array → lo forziamo
    if n_folds == 1:
        axes = [axes]

    for i in range(n_folds):

        residuals_1 = model_values_1['residuals_per_fold'][i]
        residuals_2 = model_values_2['residuals_per_fold'][i]
        year_week = model_values_1['year_week_per_fold'][i]

        x = range(len(residuals_1))
        labels = [f"{y}-W{w:02d}" for y, w in year_week]

        ax = axes[i]

        ax.plot(x, residuals_1, marker='o', linestyle='-', color='blue',
                label=model_values_1['model_name'])

        ax.plot(x, residuals_2, marker='o', linestyle='-', color='red',
                label=model_values_2['model_name'])

        ax.axhline(0, linestyle='--')
        ax.set_title(f"Fold {i+1}", fontsize=16)
        ax.set_ylabel("Residuals", fontsize=14)

        step = max(len(x) // 8, 1)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(labels[::step], rotation=45, fontsize=14)

        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=14)

    axes[-1].set_xlabel("Week (Year–Week)", fontsize=14)

    plt.tight_layout()

    fig_name = "cfr_residuals_lr_rf_all_folds.png"
    so.save_plot(fig_name, folders)

    #plt.show()


def results_table(model_values_1, model_values_2, model_values_3, folders):
    """
    Create a comparison table of model performance metrics.
    """
    results_table = pd.DataFrame({
        "Model": [
            'Linear Regression on Chronological Hold-out',
            model_values_2['model_name'],
            model_values_3['model_name']
        ],
        "MAE": [
            f"{model_values_1['mae']:.2f}",
            f"{model_values_2['mae_mean']:.2f} ± {model_values_2['mae_std']:.2f}",
            f"{model_values_3['mae_mean']:.2f} ± {model_values_3['mae_std']:.2f}"
        ],
        "RMSE": [
            f"{model_values_1['rmse']:.2f}",
            f"{model_values_2['rmse_mean']:.2f} ± {model_values_2['rmse_std']:.2f}",
            f"{model_values_3['rmse_mean']:.2f} ± {model_values_3['rmse_std']:.2f}"
        ],
        "MAPE (%)": [
            f"{model_values_1['mape']:.2f}",
            f"{model_values_2['mape_mean']:.2f} ± {model_values_2['mape_std']:.2f}",
            f"{model_values_3['mape_mean']:.2f} ± {model_values_3['mape_std']:.2f}"
        ]
    })

    print(results_table)
    so.save_table(results_table, "resuls_table_cfr_MAE_RMSE_MAPE", folders)
    return results_table