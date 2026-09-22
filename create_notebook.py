import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# House Price Prediction using Machine Learning\n",
    "**Author:** Moinuddin  \n",
    "**Project:** IBM AI & ML Internship / Industry Capstone  \n",
    "**Dataset:** House Price Prediction Dataset (`House Price Prediction Dataset.csv`)  \n",
    "\n",
    "--- \n",
    "## 1. Project Overview & Objectives\n",
    "The primary goal of this project is to build an end-to-end Machine Learning model to accurately predict residential real estate prices based on key property features such as Area, Bedrooms, Bathrooms, Floors, Year Built, Location, Condition, and Garage availability.\n",
    "\n",
    "### Workflow Steps:\n",
    "1. **Data Loading & Inspection**: Examining dataset structure, missing values, and summary statistics.\n",
    "2. **Exploratory Data Analysis (EDA)**: Visualizing distributions, feature relationships, and correlation patterns.\n",
    "3. **Data Preprocessing & Feature Encoding**: Handling categorical features (One-Hot Encoding) and standardizing numerical scales.\n",
    "4. **Model Building**: Training multiple algorithms including Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting.\n",
    "5. **Model Evaluation & Diagnostics**: Comparing model metrics ($R^2$, MAE, RMSE) and analyzing feature importances."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 2. Importing Libraries and Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "\n",
    "# Set display and plot preferences\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})\n",
    "pd.set_option('display.max_columns', None)\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('House Price Prediction Dataset.csv')\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 3. Data Inspection and Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking missing values and data types\n",
    "print(\"=== Dataset Info ===\")\n",
    "df.info()\n",
    "\n",
    "print(\"\\n=== Missing Values ===\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "print(\"\\n=== Summary Statistics ===\")\n",
    "df.describe().T\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 4. Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target Variable Distribution (Price)\n",
    "plt.figure(figsize=(9, 5))\n",
    "sns.histplot(df['Price'], kde=True, color='#2b5c8f', bins=30)\n",
    "plt.title('Target Variable Distribution: House Price', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.xlabel('Price ($)', fontsize=12)\n",
    "plt.ylabel('Frequency', fontsize=12)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Price vs Area scatter plot grouped by Location\n",
    "plt.figure(figsize=(9, 5.5))\n",
    "sns.scatterplot(data=df, x='Area', y='Price', hue='Location', style='Condition', alpha=0.8, palette='deep', s=70)\n",
    "plt.title('House Price vs. Property Area (sq ft) by Location', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.xlabel('Area (Square Feet)', fontsize=12)\n",
    "plt.ylabel('Price ($)', fontsize=12)\n",
    "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation Heatmap for Numerical Features\n",
    "df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['Id'], errors='ignore')\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(df_numeric.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)\n",
    "plt.title('Numerical Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 5. Data Preprocessing & Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate features (X) and target variable (y)\n",
    "X = df.drop(columns=['Id', 'Price'])\n",
    "y = df['Price']\n",
    "\n",
    "num_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']\n",
    "cat_cols = ['Location', 'Condition', 'Garage']\n",
    "\n",
    "# 80-20 Train-Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "print(f\"Training set size: {X_train.shape[0]} rows\")\n",
    "print(f\"Testing set size: {X_test.shape[0]} rows\")\n",
    "\n",
    "# ColumnTransformer for Preprocessing\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', StandardScaler(), num_cols),\n",
    "        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)\n",
    "    ]\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 6. Model Training & Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Linear Regression': LinearRegression(),\n",
    "    'Ridge Regression': Ridge(alpha=1.0),\n",
    "    'Lasso Regression': Lasso(alpha=1.0),\n",
    "    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=8),\n",
    "    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),\n",
    "    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])\n",
    "    pipeline.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred_train = pipeline.predict(X_train)\n",
    "    y_pred_test = pipeline.predict(X_test)\n",
    "    \n",
    "    train_r2 = r2_score(y_train, y_pred_train)\n",
    "    test_r2 = r2_score(y_test, y_pred_test)\n",
    "    mae = mean_absolute_error(y_test, y_pred_test)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))\n",
    "    \n",
    "    results[name] = {\n",
    "        'Train R2': round(float(train_r2), 4),\n",
    "        'Test R2': round(float(test_r2), 4),\n",
    "        'MAE ($)': round(float(mae), 2),\n",
    "        'RMSE ($)': round(float(rmse), 2)\n",
    "    }\n",
    "\n",
    "results_df = pd.DataFrame(results).T\n",
    "print(\"=== Model Evaluation Summary ===\")\n",
    "display(results_df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 7. Model Performance Comparison & Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visual comparison plot\n",
    "fig, ax1 = plt.subplots(figsize=(10, 5))\n",
    "x = np.arange(len(results_df))\n",
    "width = 0.35\n",
    "\n",
    "rects1 = ax1.bar(x - width/2, results_df['Test R2'], width, label='Test R² Score', color='#2b5c8f')\n",
    "ax1.set_ylabel('R² Score', fontsize=12, color='#2b5c8f')\n",
    "\n",
    "ax2 = ax1.twinx()\n",
    "rects2 = ax2.bar(x + width/2, results_df['RMSE ($)'], width, label='RMSE ($)', color='#e05d5d')\n",
    "ax2.set_ylabel('RMSE ($)', fontsize=12, color='#e05d5d')\n",
    "\n",
    "ax1.set_xticks(x)\n",
    "ax1.set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=11)\n",
    "plt.title('Machine Learning Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)\n",
    "fig.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 8. Feature Importance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Importance from Random Forest Regressor\n",
    "rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100))])\n",
    "rf_pipeline.fit(X_train, y_train)\n",
    "\n",
    "cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']\n",
    "cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()\n",
    "feature_names = num_cols + cat_feature_names\n",
    "\n",
    "importances = rf_pipeline.named_steps['model'].feature_importances_\n",
    "feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)\n",
    "\n",
    "plt.figure(figsize=(9, 5.5))\n",
    "feat_imp.plot(kind='barh', color='#2ca02c')\n",
    "plt.title('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold', pad=12)\n",
    "plt.xlabel('Relative Importance Weight', fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 9. Conclusion & Key Takeaways\n",
    "1. **Baseline Evaluation**: Linear and non-linear regression models were benchmarked on 2000 real estate samples.\n",
    "2. **Dataset Nature**: The target variable `Price` shows synthetic uniform distribution across standard feature combinations.\n",
    "3. **Pipeline Modularization**: Scikit-Learn `Pipeline` and `ColumnTransformer` guarantee clean preprocessing without data leakage.\n",
    "4. **Recommendations**: Feature engineering with domain-specific interactions (price per sq ft, age of house) and real-world geolocation data can further boost predictive standard accuracy."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('Moinuddin_HousePricePrediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Moinuddin_HousePricePrediction.ipynb created successfully!")
