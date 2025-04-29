import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)

# 1) Load & clean
df = pd.read_csv('../Data/Phase 2/final_dataset_phase2.csv')
df.rename(columns={'Customer Focused': 'Customer_Focused'}, inplace=True)
df['Sector'] = df['Sector'].str.strip()

# 1.5) Drop rows where Profit Change is missing or invalid
df = df.dropna(subset=['Profit Change'])
df = df[np.isfinite(df['Profit Change'])]

# 2) Classification setup
df['Target'] = (df['Profit Change'] > 0).astype(int)
le = LabelEncoder()
df['Sector_enc'] = le.fit_transform(df['Sector'])
joblib.dump(le, 'sector_encoder.joblib')

feature_cols = ['Profit 2022', 'Profit 2024', 'Sector_enc', 'Customer_Focused']
X = df[feature_cols]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Tune & train classifier
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid, cv=3, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)

# 4) Classification metrics
y_pred_cls = grid.predict(X_test)
y_prob_cls = grid.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_cls):.4f}")
print(f"Test ROC AUC : {roc_auc_score(y_test, y_prob_cls):.4f}")

joblib.dump(grid.best_estimator_, 'xgb_model.joblib')

# ——————————————————————————————

# 5) Regression on Profit Change
y_reg = df['Profit Change']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    random_state=42
)
reg.fit(Xr_train, yr_train)
y_pred = reg.predict(Xr_test)

# 6) Regression metrics
rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
mae  = mean_absolute_error(      yr_test, y_pred)
mape = np.mean(np.abs((yr_test - y_pred) / yr_test)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 7) Plot Actual vs Predicted
plt.figure()
plt.scatter(yr_test, y_pred, alpha=0.6)
plt.xlabel("Actual Profit Change")
plt.ylabel("Predicted Profit Change")
plt.title("Actual vs. Predicted Profit Change")
plt.tight_layout()
plt.show()

# 8) Plot Residuals
residuals = yr_test - y_pred
plt.figure()
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel("Residual (Actual − Predicted)")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()
