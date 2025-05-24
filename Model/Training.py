import os
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

# ─── Setup output dir ─────────────────────────────────────────────────────────
os.makedirs('../outputs', exist_ok=True)

# 1) Load & clean
df = pd.read_csv('../Data/Phase 2/final_dataset_phase2.csv')
df.rename(columns={'Customer Focused': 'Customer_Focused'}, inplace=True)
df['Sector'] = df['Sector'].str.strip()
# Drop bad labels
df = df.dropna(subset=['Profit Change'])
df = df[np.isfinite(df['Profit Change'])]

# ─── Classification ────────────────────────────────────────────────────────────
df['Target'] = (df['Profit Change'] > 0).astype(int)
le = LabelEncoder()
df['Sector_enc'] = le.fit_transform(df['Sector'])
joblib.dump(le, 'sector_encoder.joblib')

features = ['Profit 2022', 'Profit 2024', 'Sector_enc', 'Customer_Focused']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    {'n_estimators': [100, 200], 'max_depth': [3, 5]},
    cv=3, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)

y_pred_cls = grid.predict(X_test)
y_prob_cls = grid.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred_cls)
auc = roc_auc_score(y_test, y_prob_cls)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC AUC : {auc:.4f}")

joblib.dump(grid.best_estimator_, 'xgb_model.joblib')

# ─── Regression ────────────────────────────────────────────────────────────────
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

rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
mae  = mean_absolute_error(      yr_test, y_pred)
mape = np.mean(np.abs((yr_test - y_pred) / yr_test)) * 100
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# ─── Save combined metrics ─────────────────────────────────────────────────────
metrics_txt = (
    f"Test Accuracy: {acc:.4f}\n"
    f"Test ROC AUC : {auc:.4f}\n"
    f"RMSE         : {rmse:.2f}\n"
    f"MAE          : {mae:.2f}\n"
    f"MAPE         : {mape:.2f}%\n"
)
with open('../outputs/metrics.txt', 'w') as f:
    f.write(metrics_txt)

# ─── Plot #1: Zoomed‐in Scatter ─────────────
# Scale to billions
yr_b = yr_test.values / 1e9
yp_b = y_pred           / 1e9

# Compute 1st/99th percentiles to zoom
all_vals = np.concatenate([yr_b, yp_b])
ymin, ymax = np.percentile(all_vals, [1, 99])

plt.figure(figsize=(8, 5))
plt.scatter(yr_b, yp_b, alpha=0.7, edgecolor='k')
plt.plot([ymin, ymax], [ymin, ymax], 'r--', linewidth=1)  # 45° line
plt.ylim(ymin, ymax)
plt.xlim(ymin, ymax)
plt.xlabel("Actual Profit Change (Billion USD)")
plt.ylabel("Predicted Profit Change (Billion USD)")
plt.title("Zoomed: Actual vs. Predicted Profit Change")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../outputs/actual_vs_pred_zoomed_scatter.png')
plt.close()

# ─── Plot #2 (version 10) FIXED: Filtered Scatter with Optional Lines ────────────────────────
# Scale to billions
yr_b = yr_test.values / 1e9
yp_b = y_pred           / 1e9

# Build results DataFrame
results = pd.DataFrame({
    'Actual':    yr_b,
    'Predicted': yp_b
}).reset_index(drop=True)

# Define zoom window
low, high = np.percentile(np.concatenate([yr_b, yp_b]), [1, 99])

# Filter to the normal range
mask = (results['Actual'] >= low) & (results['Actual'] <= high) & \
       (results['Predicted'] >= low) & (results['Predicted'] <= high)
zoomed = results[mask].reset_index(drop=True)

plt.figure(figsize=(8, 5))
# Scatter only (no vertical line artefacts)
plt.scatter(zoomed.index+1, zoomed['Actual'],
            marker='o', label='Actual', alpha=0.7)
plt.scatter(zoomed.index+1, zoomed['Predicted'],
            marker='o', label='Predicted', alpha=0.7)

plt.plot(zoomed.index+1, zoomed['Actual'],    linestyle='-', linewidth=0.5, alpha=0.3)
plt.plot(zoomed.index+1, zoomed['Predicted'], linestyle='-', linewidth=0.5, alpha=0.3)

plt.xlim(zoomed.index.min(), zoomed.index.max())
plt.ylim(low, high)
plt.xlabel('Test Sample #')
plt.ylabel('Profit Change (Billion USD)')
plt.title('Zoomed: Actual vs Predicted (Filtered Scatter)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../outputs/line_actual_vs_pred_zoomed_fixed.png')
plt.show()
plt.close()


# ─── Plot #3: Filtered Residuals Histogram ────────────────────────────────────
res_b = (yr_test - y_pred) / 1e9

low, high = np.percentile(res_b, [1, 99])

res_filtered = res_b[(res_b >= low) & (res_b <= high)]

plt.figure(figsize=(8, 4))
plt.hist(
    res_filtered,
    bins=20,
    edgecolor='k',
    alpha=0.7,
    range=(low, high)            # ensure bins only cover the filtered range
)
plt.xlabel("Residual (Actual − Predicted) (Billion USD)")
plt.title("Filtered Residual Distribution (1st–99th percentile)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../outputs/residuals_filtered.png')
plt.show()
plt.close()


# im so tired of this bs