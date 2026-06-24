import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend alternativo, funziona con finestre grafiche
from src import data_cleaning as dc
from src import data_preparation as dp
from src import exploratory_data_analysis as eg
from src import feature_engineering as fe
from src import forecasting as fc
from src import modelling_and_validation_strategies as mvs
from src import final_model_explanation as sa
from src import saving_output as so
from src import final_model_evaluation as me
from src import model_selection as ms


def main():

    #=================
    #DATA PREPARATION
    #=================

    #Create structured output folders for the current run (graphs, tables, datasets, and text outputs)
    folders=so.create_output_folders()


    #Load datasets from csv files
    features, stores, test, train = dp.loading_datasets()

    #GENERAL INFORMATION
    #Save general information about all datasets
    so.save_text("\n\n\n\n ---- GENERAL INFORMATION REGARDING EACH INITIAL DATASET ----- \n", "general_info", folders)

    #feature dataset
    so.save_text("\n--> features.csv\n", "general_info", folders)
    dp.general_info(features, 'Store', folders)

    #store dataset
    so.save_text("\n--> stores.csv\n", "general_info", folders)
    dp.general_info(stores, 'Store', folders)

    #test dataset
    so.save_text("\n--> test.csv\n", "general_info", folders)
    dp.general_info(test, 'Store', folders)

    #train dataset
    so.save_text("\n--> train.csv\n", "general_info", folders)
    dp.general_info(train, 'Store', folders)

    #STORE SELECTION
    # Find the store with the lowest number of missing values for each dataset
    so.save_text("----BEST STORE BY MISSING VALUES----", "final_dataset_creation", folders)

    #feature dataset
    so.save_text("\n--> features.csv\n", "final_dataset_creation", folders)
    features_missing_per_store, features_best_store = dp.best_store_by_missing(features, 'Store', folders)

    #stores dataset
    so.save_text("\n--> stores.csv\n", "final_dataset_creation", folders)
    stores_missing_per_store, stores_best_store = dp.best_store_by_missing(stores, 'Store', folders)

    #test dataset
    so.save_text("\n--> test.csv\n", "final_dataset_creation", folders)
    test_missing_per_store, test_best_store = dp.best_store_by_missing(test, 'Store', folders)

    #train dataset
    so.save_text("\n--> train.csv\n", "final_dataset_creation", folders)
    train_missing_per_store, train_best_store = dp.best_store_by_missing(train, 'Store', folders)

    #Focus on store 20 for further analysis
    features_20 = dp.new_database_20(features, 'Store', 20)
    stores_20 = dp.new_database_20(stores, 'Store', 20)
    train_20 = dp.new_database_20(train, 'Store', 20)
    test_20 = dp.new_database_20(test, 'Store', 20)


    # Consistency check on filtered datasets (store 20)
    print('Features:\n')
    dp.check_df_store_20(features_20, 'Store')

    print('\n \n Stores:\n')
    dp.check_df_store_20(stores_20, 'Store')

    print('\n \n Train:\n')
    dp.check_df_store_20(train_20, 'Store')

    print('\n \n Test:\n')
    dp.check_df_store_20(test_20, 'Store')

    #DEPARTMENT SELECTION
    # Find the department with the lowest number of missing values (store 20)
    so.save_text("\n\n----BEST DEPARTMENT BY MISSING VALUES IN STORE 20----", "final_dataset_creation", folders)

    #train dataset
    so.save_text("\n--> train_20.csv\n", "final_dataset_creation", folders)
    missing_per_dept_train, best_dept_train = dp.best_dept_by_missing(train_20, 'Dept', folders)

    #test dataset
    so.save_text("\n--> test_20.csv\n", "final_dataset_creation", folders)
    missing_per_dept_test, best_dept_test = dp.best_dept_by_missing(test_20, 'Dept', folders)


    # Find the department with the highest number of rows (store 20)
    so.save_text("\n\n----BEST DEPARTMENT BY NUMBER OF ROWS IN STORE 20----", "final_dataset_creation", folders)

    #train dataset
    print('Train: ')
    so.save_text("\n--> train_20.csv\n", "final_dataset_creation", folders)
    best_dept_train = dp.best_dept_by_rows(train_20, 'Dept', folders)

    #test dataset
    print('Test: ')
    so.save_text("\n--> test_20.csv\n", "final_dataset_creation", folders)
    best_dept_test = dp.best_dept_by_rows(test_20, 'Dept', folders)

    #DATASET FILTERING AND MERGING
    #Focus on store 20 and department 1
    print("train_20_1 \n")
    train_20_1 = dp.new_database_20_1(train_20, 'Store', 'Dept', 1)

    print("\ntest_20_1 \n")
    test_20_1 = dp.new_database_20_1(test_20, 'Store', 'Dept', 1)

    #Check columns of the new datasets
    print(
        f'train_20_1 columns: \n{train_20_1.columns} \n \ntest_20_1 columns: '
        f'\n{test_20_1.columns}\n \nfeatures_20 columns: \n {features_20.columns}')

    #Convert Date column to datetime format (if needed)
    train_20_1=dp.dates_type_check(train_20_1)
    test_20_1=dp.dates_type_check(test_20_1)
    features_20=dp.dates_type_check(features_20)

    #Plot weekly sales over time (train set)
    eg.weekly_sales_x_week(train_20_1,folders)

    #Merge datasets to create final train and test sets
    train_full_20_1=dp.merging_datasets(train_20_1, features_20)
    test_full_20_1=dp.merging_datasets(test_20_1, features_20)

    #Final check of train and test datasets before modeling
    print(f'train shape: {train_full_20_1.shape} \n'
          f'\ntest shape: {test_full_20_1.shape} \n'
          f'\ntrain missing values: \n'
          f'{train_full_20_1.isna().sum()} \n'
          f'\n\ntest missing values: \n{test_full_20_1.isna().sum()} \n'
          f'\n\ntrain columns: \n{train_full_20_1.columns} \n'
          f'\n\ntest columns: \n{test_full_20_1.columns}'
          )

    #Drop unnecessary columns before modeling
    train_full_20_1 = train_full_20_1.drop(columns=['Store', 'Dept', 'missing_in_row_x', 'missing_in_row_y'])
    test_full_20_1 = test_full_20_1.drop(columns=['Store', 'Dept', 'missing_in_row_x', 'missing_in_row_y'])

    # Check final dataset columns after preprocessing
    print(f'New train_full_20_1 database columns: '
        f'{train_full_20_1.columns} \n\n '
        f'New test_full_20_1 database columns: {test_full_20_1.columns}')

    # =================
    # DATA CLEANING
    # =================

    #HOLIDAY DATES CONGRUENCE ANALYSIS
    # Check consistency of holiday information across datasets
    # Reference holiday dates
    holiday_dates = ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08', '2010-09-10', '2011-09-09', '2012-09-07',
                     '2013-09-06', '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29', '2010-12-31', '2011-12-30',
                     '2012-12-28', '2013-12-27']
    holiday_dates = pd.to_datetime(holiday_dates)

    print('DATABASE train_full_20_1 \n')
    errors_train_full_20_1 = dc.check_holiday_dates(train_full_20_1, holiday_dates)
    print(f'\n errors in the dates of the database train_full_20_1: \n{errors_train_full_20_1} \n\n')

    print('DATABASE test_full_20_1 \n')
    errors_test_full_20_1 = dc.check_holiday_dates(test_full_20_1, holiday_dates)
    print(f'\n errors in the dates of the database test_full_20_1: \n{errors_test_full_20_1} \n\n')

    # Remove redundant columns and standardize holiday column name
    test_full_20_1.columns

    print('I remove redundant columns and change the header of IsHoliday_x. \n \n')
    print('train_full_20_1')
    train_full_20_1 = train_full_20_1.drop(columns=['IsHoliday_y', 'IsHoliday_list', 'Holiday_mismatch'])
    train_full_20_1 = train_full_20_1.rename(columns={'IsHoliday_x': 'IsHoliday'})
    print(f'columns of train_full_20_1: {train_full_20_1.columns} \n\n')

    print('test_full_20_1')
    test_full_20_1 = test_full_20_1.drop(columns=['IsHoliday_y', 'IsHoliday_list', 'Holiday_mismatch'])
    test_full_20_1 = test_full_20_1.rename(columns={'IsHoliday_x': 'IsHoliday'})
    print(f'columns of test_full_20_1: {test_full_20_1.columns}')

    #MISSING DATA TREATMENT
    # Check missing values in final datasets
    dc.n_missing_values(train_full_20_1)
    dc.n_missing_values(test_full_20_1)

    # Fill missing values in markdown features using 0
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    train_full_20_1=dc.missing_values_imputation(train_full_20_1, markdown_cols, method=None, value=0)
    test_full_20_1 = dc.missing_values_imputation(test_full_20_1, markdown_cols, method=None, value=0)

    # Impute missing values in macroeconomic variables using forward fill
    macro_cols = ['CPI', 'Unemployment']
    test_full_20_1=dc.missing_values_imputation(test_full_20_1, macro_cols, method='ffill', value=None)

    # Check missing values after treatment
    dc.n_missing_values(test_full_20_1)
    dc.n_missing_values(train_full_20_1)


    #OUTLIER ANALYSIS
    # Analyze distribution of weekly sales
    dc.weekly_sales_distribution_boxplot(train_full_20_1, folders)

    # Compare weekly sales distribution between holiday and non-holiday dates
    dc.weekly_sales_distribution_by_holiday(train_full_20_1, folders)

    # Create holiday period feature using rolling window
    train_full_20_1, test_full_20_1=fe.is_holiday_period_variable(train_full_20_1, test_full_20_1)
    # Analyze weekly sales distribution by holiday period
    dc.weekly_sales_distribution_by_holiday_period(train_full_20_1, folders)

    # Compare feature behavior between normal observations and sales outliers
    dc.feature_averages_comparison_overall_vs_sales_outliers(train_full_20_1, folders)

    # =================
    # EXPLORATORY DATA ANALYSIS
    # =================

    # Explore relationship between features and weekly sales
    eg.plotting_basic(train_full_20_1, 'Temperature', folders)
    eg.plotting_basic(train_full_20_1, 'Fuel_Price', folders)
    eg.plotting_basic(train_full_20_1, 'MarkDown1', folders)
    eg.plotting_basic(train_full_20_1, 'MarkDown2', folders)
    eg.plotting_basic(train_full_20_1, 'MarkDown3', folders)
    eg.plotting_basic(train_full_20_1, 'MarkDown4', folders)
    eg.plotting_basic(train_full_20_1, 'MarkDown5', folders)
    eg.plotting_basic(train_full_20_1, 'CPI', folders)
    eg.plotting_basic(train_full_20_1, 'Unemployment', folders)
    eg.avg_weekly_sales_by_week_type(train_full_20_1, folders)


    # =================
    # FEATURE ENGINEERING
    # =================

    # Create calendar-based features from date information
    train_full_20_1=fe.calendar_features(train_full_20_1)
    test_full_20_1=fe.calendar_features(test_full_20_1)

    # Analyze seasonal patterns in weekly sales over different time periods
    eg.avg_weekly_sales_by_time_period(train_full_20_1,'Year', folders)
    eg.avg_weekly_sales_by_time_period(train_full_20_1,'Month', folders)
    eg.avg_weekly_sales_by_time_period(train_full_20_1,'Week', folders)

    # Sort data by date to ensure correct time order
    train_full_20_1 = train_full_20_1.sort_values('Date')
    train_full_20_1.head()

    # Create lag and rolling mean features for time series modeling
    train_full_20_1, test_full_20_1=fe.lag_features_roll_mean_features(train_full_20_1, test_full_20_1)

    # Final check for missing values after feature engineering
    dc.n_missing_values(train_full_20_1)
    dc.n_missing_values(test_full_20_1)

    # Final imputation of remaining missing values before modeling
    train_full_20_1 = train_full_20_1.fillna(0)
    test_full_20_1 = test_full_20_1.fillna(0)

    # Final sanity check before model training
    train_full_20_1.columns
    train_full_20_1.head()
    test_full_20_1.columns
    test_full_20_1.head()
    dc.n_missing_values(train_full_20_1)
    dc.n_missing_values(test_full_20_1)

    # =================
    # MODELLING AND VALIDATION STRATEGY
    # =================

    #TRAIN/TEST SPLIT
    # Separate features and target variable for model training
    X, y =mvs.features_target_separation(train_full_20_1)

    # Check feature and target shapes before model training
    print(X.shape)
    print(X.columns)

    print(y.shape)
    print(y.name)

    #VALIDATION STRATEGY DEFINITION
    #CHRONOLOGICAL HOLD-OUT VALIDATION
    # Split data into training and validation sets using chronological order
    X_train_50, X_val_50, y_train_50, y_val_50 = mvs.chronological_hold_out_validation(X, y)

    #TIMESERIESSPLIT CROSS-VALIDATION
    # Define time series cross-validation strategy
    tscv=mvs.timeSeriesSplit_cross_validation()

    #MODEL TRAINING AND METRIC COMPUTATION
    #LINEAR REGRESSION ON 50/50 HOLD-OUT
    # Train Linear Regression model on 50/50 chronological split and evaluate performance
    lr, y_pred_50=mvs.lr_chronological_hold_out(X_train_50, y_train_50, X_val_50, y_val_50)
    lr_hold_out_values=mvs.mae_rmse_mape_calculation(y_val_50, y_pred_50)

    #LINEAR REGRESSION ON TIME SERIES SPLIT CV
    # Evaluate Linear Regression using Time Series Cross-Validation
    lr_TSS_model_values=mvs.output_strings_TimeSeriesSplit('Linear Regression on TimeSeriesSplit')
    lr_TSS_model_values=mvs.model_fit_TimeSeriesSplit(lr, tscv, lr_TSS_model_values,X,y,folders)
    lr_TSS_model_values=mvs.avg_mae_rmse_mape(lr_TSS_model_values)
    so.model_values_to_df(lr_TSS_model_values, folders)

    # Analyze model performance using residuals and predicted vs actual weekly sales comparison (graphs)
    ms.residual_analysis(lr_TSS_model_values, folders)
    ms.comparison_predicted_actual_weekly_sales_TSS(lr_TSS_model_values, folders)

    #RANDOM FOREST ON TIME SERIES SPLIT CV
    # Train and evaluate Random Forest model using TimeSeriesSplit cross-validation
    rf=mvs.rf_definition()
    rf_TSS_model_values=mvs.output_strings_TimeSeriesSplit('Random Forest on TimeSeriesSplit')
    rf_TSS_model_values = mvs.model_fit_TimeSeriesSplit(rf, tscv, rf_TSS_model_values,X,y, folders)
    rf_TSS_model_values = mvs.avg_mae_rmse_mape(rf_TSS_model_values)
    so.model_values_to_df(rf_TSS_model_values, folders)

    # Analyze model performance using residuals and predicted vs actual weekly sales comparison (graphs)
    ms.residual_analysis(rf_TSS_model_values, folders)
    ms.comparison_predicted_actual_weekly_sales_TSS(rf_TSS_model_values, folders)

    # =================
    # MODEL SELECTION
    # =================

    # Compare model performance using MAE, RMSE, and MAPE
    results_table=ms.results_table(lr_hold_out_values, lr_TSS_model_values, rf_TSS_model_values, folders)

    # Compare Linear Regression and Random Forest predictions and residuals
    ms.model_comparison_lr_rf(lr_TSS_model_values, rf_TSS_model_values, folders)
    ms.residual_analysis_comparison_lr_rf(lr_TSS_model_values, rf_TSS_model_values, folders)

    # =================
    # FORECASTING
    # =================

    #FINAL DATASET CREATION AND FINAL MODEL TRAINING
    # Prepare final datasets for training and forecasting
    X_train_final, y_train_final, X_test=fc.final_forecasting_datasets(train_full_20_1, test_full_20_1)
    print(f'{X_train_final.columns} \n {X_train_final.shape}')
    print(f'{y_train_final.name} \n {y_train_final.shape}')
    print(f'{X_test.columns} \n {X_test.shape}')

    # Define final Random Forest model for forecasting
    final_model=fc.final_model_rf_definition()

    #TEST PREDICTION
    # Train final model and generate forecasts on test data
    test_full_20_1=fc.final_model_forecasting(X_train_final, y_train_final, final_model, test_full_20_1, X_test)
    so.save_dataset(test_full_20_1, "test_full_20_1_with_predictions", folders)

    #FORECASTED WEEKLY SALES PLOT
    # Plot final forecasted weekly sales over time
    fc.forecasted_weekly_sales_plot(test_full_20_1, folders)

    # =================
    # FINAL MODEL EVALUATION
    # =================

    # Compare distributions and test similarity between real and predicted sales
    me.historical_predicted_weekly_sales_distributions_box_plot(y_train_final,test_full_20_1, folders)
    me.historical_predicted_weekly_sales_distribution_histogram(y_train_final,test_full_20_1, folders)

    #Kolomogorov-Smirnov test to evaluate similarity between historical and predicted sales distributions
    ks_stat, p_value=me.kolomogorov_smirnov_test(y_train_final, test_full_20_1, folders)

    # =================
    # FINAL MODEL EXPLANATION
    # =================

    # Compute Random Forest feature importances
    random_forest_importances = sa.importances_calculation(final_model, X_train_final, folders)

    #SHAP ANALYSIS
    # Perform SHAP analysis on a sample of training data
    X_shap, X_sample, explainer, exp, mean_abs_shap, n_sample=sa.shap_analysis(X_train_final, final_model)

    # Compute and save SHAP feature importance summary
    shap_importance=sa.shap_importance_display(X_sample,mean_abs_shap, n_sample, folders)

    # Plot SHAP feature importance (bar and beeswarm plots)
    sa.shap_summary_plot(exp, X_sample, folders)
    # Plot SHAP waterfall explanation for a single prediction
    sa.shap_summary_plot_waterfall(exp, folders, i=0)

if __name__ == "__main__":
    main()