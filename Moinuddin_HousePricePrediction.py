"""
=============================================================================
Project Title : House Price Prediction using Machine Learning
Author        : Moinuddin
Dataset       : House Price Prediction Dataset.csv
Description   : Complete Machine Learning Pipeline for predicting house prices
                including EDA, Data Preprocessing, Feature Engineering, 
                Model Training, Evaluation, and Performance Visualization.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set visualization settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})


def load_dataset(file_path):
    """Load and perform initial data check."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
    
    df = pd.read_csv(file_path)
    print("=" * 60)
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print("=" * 60)
    print("\nData Preview:")
    print(df.head())
    print("\nData Info:")
    df.info()
    return df


def perform_eda(df, output_dir="figures"):
    """Perform Exploratory Data Analysis and save plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Price'], kde=True, color='#2b5c8f', bins=30)
    plt.title('Target Variable Distribution: House Price', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_price_dist.png'), dpi=300)
    plt.close()

    # 2. Area vs Price Scatter Plot
    plt.figure(figsize=(9, 5.5))
    sns.scatterplot(data=df, x='Area', y='Price', hue='Location', style='Condition', alpha=0.8, palette='deep', s=70)
    plt.title('House Price vs. Property Area (sq ft)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Area (Square Feet)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_area_price.png'), dpi=300)
    plt.close()

    # 3. Correlation Matrix
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Id'], errors='ignore')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title('Numerical Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation.png'), dpi=300)
    plt.close()

    print(f"\nEDA Visualizations successfully saved to '{output_dir}/' directory.")


def build_and_evaluate_models(df):
    """Train multiple regression models and evaluate performance."""
    # Drop Id feature
    df_clean = df.drop(columns=['Id'], errors='ignore')
    
    X = df_clean.drop(columns=['Price'])
    y = df_clean['Price']

    num_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
    cat_cols = ['Location', 'Condition', 'Garage']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=8),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    }

    results = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        results[name] = {
            'Train R2': round(float(train_r2), 4),
            'Test R2': round(float(test_r2), 4),
            'MAE ($)': round(float(mae), 2),
            'RMSE ($)': round(float(rmse), 2)
        }

        print(f"\n[{name}]")
        print(f"  - Train R² Score : {train_r2:.4f}")
        print(f"  - Test R² Score  : {test_r2:.4f}")
        print(f"  - MAE ($)        : ${mae:,.2f}")
        print(f"  - RMSE ($)       : ${rmse:,.2f}")

    return pd.DataFrame(results).T, X_train, y_train, preprocessor


def main():
    dataset_file = 'House Price Prediction Dataset.csv'
    
    # 1. Load Data
    df = load_dataset(dataset_file)

    # 2. EDA
    perform_eda(df)

    # 3. Model Training & Evaluation
    results_df, X_train, y_train, preprocessor = build_and_evaluate_models(df)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string())

    print("\nPipeline execution finished successfully.")


if __name__ == '__main__':
    main()
