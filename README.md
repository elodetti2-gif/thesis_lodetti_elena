# Demand Forecasting for Retail Store Using Machine Learning (with Python)

## Project Overview
This thesis project focuses on weekly demand forecasting for a retail store using machine learning techniques.  
The goal is to support inventory management and operational decision-making by building interpretable and predictive models based on historical sales data.

The analysis is based on the "Walmart Recruiting – Store Sales Forecasting" dataset and considers a single store–department combination to ensure interpretability and methodological clarity.

---

## Methodology
The project follows a complete machine learning pipeline structured into the following stages:

- Data preparation and filtering of raw datasets
- Data cleaning, including missing value treatment and outlier analysis
- Exploratory Data Analysis (EDA) with time series visualization and feature–target relationships
- Feature engineering, including:
  - calendar-based features (week, month, year)
  - lag features
  - rolling statistics
  - holiday indicators
- Model development and validation using time-aware strategies:
  - chronological train/test split
  - TimeSeriesSplit cross-validation
- Model selection and evaluation using:
  - Linear Regression (baseline model)
  - Random Forest Regressor (non-linear model)
  - metrics: MAE, RMSE, MAPE
- Forecast generation and post-analysis
- Distributional analysis of residuals (including Kolmogorov–Smirnov test)
- Model interpretation using SHAP values and feature importance

---

## Project Structure
```text
data/  #Raw input datasets
├── features.csv  #Store-level and regional conditions
├── stores.csv  #Store information
├── train.csv  #Historical sales data for training
├── test.csv  #Data for sales forecasting

src/
├── data_preparation.py  #Data loading, filterning and merging
├── data_cleaning.py  #Consistency checks, missing data handling and outlier analysis
├── exploratory_data_analysis.py  #Data exploration and descriptive analysis
├── feature_engineering.py  #Feature generation, transformation and selection
├── modelling_and_validation_strategy.py  #Data preparation and validation strategy definition
├── model_selection.py  #Model training, evaluation, comparison and final selection
├── forecasting.py  #Sales forecasting on the test dataset
├── final_model_evaluation.py  #Final performance assessment
├── final_model_explanation.py #Model interpretation and feature importance analysis
├── saving_output.py #Output generation and storage
├── __init__.py

output/  #Generated outputs
├── run_id #Timestamped folder for each execution
│   ├── dataset/ #Processed datasets (.csv)
│   ├── graph/ #Plots and visualizations (.png)
│   ├── table/ #Excel result tables (.xlsx)
│   ├── text/  #Logs and textual reports (.txt)

main.py  #Main pipeline orchestrator
requirements.txt  #Project dependencies
README.md  #This file
```

## Installation
Install the required dependencies using:

pip install -r requirements.txt

Main libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib
- shap

To run the full pipeline:

python main.py

This will execute the complete workflow from data preprocessing to model evaluation and output generation.

## Outputs

Each execution generates a timestamped folder inside output/, containing four folders:

- dataset/: containing the processed dataset and predictions
- graph/: containing EDA plots, forecast visualizations, SHAP plots...
- table/: containing performance metrics, feature importances, results tables...
- text/: containing analytical summaries

### Author
Elena Lodetti
