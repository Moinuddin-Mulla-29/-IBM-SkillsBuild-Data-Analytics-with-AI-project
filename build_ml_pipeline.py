import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

os.makedirs('figures', exist_ok=True)

# 1. Load Data
df = pd.read_csv('House Price Prediction Dataset.csv')

print("Data Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop Id column for ML
df_ml = df.drop(columns=['Id'])

# Separate Features and Target
X = df_ml.drop(columns=['Price'])
y = df_ml['Price']

num_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
cat_cols = ['Location', 'Condition', 'Garage']

# 2. Exploratory Data Analysis & Plots
# Figure 1: Price Distribution
plt.figure(figsize=(9, 5))
sns.histplot(df['Price'], kde=True, color='#2b5c8f', bins=30)
plt.title('Target Variable Distribution: House Price', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('figures/eda_price_dist.png', dpi=300)
plt.close()

# Figure 2: Area vs Price
plt.figure(figsize=(9, 5.5))
sns.scatterplot(data=df, x='Area', y='Price', hue='Location', style='Condition', alpha=0.8, palette='deep', s=70)
plt.title('House Price vs. Property Area (sq ft) by Location', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Area (Square Feet)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/eda_area_price.png', dpi=300)
plt.close()

# Figure 3: Correlation Matrix
df_numeric = df_ml.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Numerical Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/eda_correlation.png', dpi=300)
plt.close()

# Figure 4: Price by Location & Condition
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Location', y='Price', hue='Condition', palette='Set2')
plt.title('Price Distribution across Location and Property Condition', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(title='Condition', loc='upper right')
plt.tight_layout()
plt.savefig('figures/eda_location_condition.png', dpi=300)
plt.close()

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

# 4. Model Training & Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=8),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
}

results = {}

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
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2)
    }

print("Model Evaluation Results:")
print(json.dumps(results, indent=2))

# Save results JSON
with open('figures/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Figure 5: Model Comparison Chart
res_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(res_df))
width = 0.35

rects1 = ax1.bar(x - width/2, res_df['Test R2'], width, label='Test R² Score', color='#2b5c8f')
ax1.set_ylabel('R² Score', fontsize=12, color='#2b5c8f')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, res_df['RMSE'], width, label='RMSE ($)', color='#e05d5d')
ax2.set_ylabel('RMSE ($)', fontsize=12, color='#e05d5d')

ax1.set_xticks(x)
ax1.set_xticklabels(res_df['Model'], rotation=15, ha='right', fontsize=11)
plt.title('Machine Learning Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)
fig.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300)
plt.close()

# Figure 6: Feature Importance from Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100))])
rf_pipeline.fit(X_train, y_train)

cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + cat_feature_names

importances = rf_pipeline.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(9, 5.5))
feat_imp.plot(kind='barh', color='#2ca02c')
plt.title('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Relative Importance Weight', fontsize=12)
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300)
plt.close()

print("ML pipeline executed successfully and figures generated.")
