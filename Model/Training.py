import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load the cleaned dataset from Phase 2
df = pd.read_csv('../Data/Phase 2/final_dataset_phase2.csv')

# 1.1 Rename column so it matches the feature list
df.rename(columns={'Customer Focused': 'Customer_Focused'}, inplace=True)

# 2. Prepare features (X) and target (y)
df['Target'] = (df['Profit Change'] > 0).astype(int)

# 2.1 Encode 'Sector' into numeric values
le = LabelEncoder()
df['Sector_enc'] = le.fit_transform(df['Sector'])
joblib.dump(le, 'sector_encoder.joblib')

# 2.2 Select feature columns
feature_cols = ['Profit 2022', 'Profit 2024', 'Sector_enc', 'Customer_Focused']
X = df[feature_cols]
y = df['Target']

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5]
}
grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)

# 5. Evaluate on the test set
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]
print('Test Accuracy:', accuracy_score(y_test, y_pred))
print('Test ROC AUC :', roc_auc_score(y_test, y_prob))

# 6. Save the trained model
joblib.dump(grid.best_estimator_, 'xgb_model.joblib')
