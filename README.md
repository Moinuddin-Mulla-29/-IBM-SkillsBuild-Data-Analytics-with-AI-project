# House Price Prediction - Machine Learning Project

![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/Status-Completed-green.svg)

---

## 📌 Project Overview
This project focuses on building an end-to-end Machine Learning pipeline to predict residential house prices based on physical property characteristics and qualitative features. The goal is to perform Exploratory Data Analysis (EDA), implement clean data preprocessing workflows using Scikit-Learn `Pipeline` and `ColumnTransformer`, train multiple regression algorithms, evaluate performance metrics ($R^2$, MAE, RMSE), and deliver comprehensive project documentation.

---

## 📊 Dataset Information
- **Dataset File Name**: [`House Price Prediction Dataset.csv`](./House%20Price%20Prediction%20Dataset.csv)
- **Total Samples**: 2,000 records
- **Total Attributes**: 10 columns (9 input features + 1 target variable `Price`)

### Data Dictionary
| Column Name | Data Type | Feature Type | Description |
| :--- | :--- | :--- | :--- |
| `Id` | Integer | Identifier | Unique identification number (dropped during ML training). |
| `Area` | Integer | Numerical | Property living area in square feet (501 - 4,999 sq ft). |
| `Bedrooms` | Integer | Numerical | Total bedroom count (1 - 5). |
| `Bathrooms` | Integer | Numerical | Total bathroom count (1 - 4). |
| `Floors` | Integer | Numerical | Number of floors (1 - 3). |
| `YearBuilt` | Integer | Numerical | Year of construction (1900 - 2023). |
| `Location` | Categorical | Discrete | Neighborhood location (`Downtown`, `Suburban`, `Urban`, `Rural`). |
| `Condition` | Categorical | Discrete | Qualitative property rating (`Poor`, `Fair`, `Good`, `Excellent`). |
| `Garage` | Categorical | Binary | Garage availability (`Yes` / `No`). |
| `Price` | Integer | Target | Property sale price in USD ($50,005 - $999,656). |

---

## 🛠️ Technologies Used
- **Programming Language**: Python 3.13+
- **Data Manipulation & Analysis**: `Pandas`, `NumPy`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Machine Learning Library**: `Scikit-Learn`
- **Document Generation**: `python-docx`
- **Development Environment**: Jupyter Notebook (`.ipynb`), VS Code / Antigravity IDE

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have Python installed on your system.

### 2. Install Required Dependencies
Run the following command in your terminal/command prompt to install all required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline Script
To run the automated Python ML script:
```bash
python Moinuddin_HousePricePrediction.py
```

### 4. Run the Jupyter Notebook
Launch Jupyter Notebook to view interactive visualizations and markdown notes:
```bash
jupyter notebook Moinuddin_HousePricePrediction.ipynb
```

---

## 📁 Project Structure & Deliverables

```
ibm-project/
│
├── House Price Prediction Dataset.csv    # Original Dataset
├── Moinuddin_HousePricePrediction.ipynb  # Main Jupyter Notebook Submission Code
├── Moinuddin_HousePricePrediction.py     # Runnable Python Pipeline Script
├── Moinuddin_ProjectReport.docx          # Complete Project Documentation (.docx)
├── requirements.txt                      # Project Dependencies List
├── README.md                             # Project Documentation & Overview
│
├── figures/                              # Generated EDA & Model Evaluation Plots
│   ├── eda_price_dist.png
│   ├── eda_area_price.png
│   ├── eda_correlation.png
│   ├── eda_location_condition.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── build_ml_pipeline.py                  # Pipeline Execution Utility Script
├── build_word_report.py                  # Automated Word Report Generator Script
└── create_notebook.py                    # Automated Notebook Generator Script
```

---

## 📈 Model Performance Summary

The following regression models were trained and benchmarked on an 80/20 train-test split:

| Algorithm Name | Train $R^2$ | Test $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.0099 | -0.0067 | $243,241.98 | $279,859.73 |
| **Ridge Regression** | 0.0099 | -0.0067 | $243,244.09 | $279,859.47 |
| **Lasso Regression** | 0.0099 | -0.0067 | $243,242.05 | $279,859.68 |
| **Gradient Boosting** | 0.1998 | -0.0355 | $245,284.36 | $283,830.91 |
| **Random Forest** | 0.8468 | -0.0984 | $252,671.95 | $292,329.63 |
| **Decision Tree** | 0.2513 | -0.2503 | $264,783.70 | $311,890.40 |

---

## 👤 Author Information
- **Name**: Moinuddin
- **Project**: House Price Prediction Machine Learning Project
