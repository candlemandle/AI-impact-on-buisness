import pandas as pd
import joblib

# 1. Load the trained model and encoder
model = joblib.load('xgb_model.joblib')
le = joblib.load('sector_encoder.joblib')

# 2. Read the new data
new_df = pd.read_csv('../Data/new_companies.csv')
new_df['Sector'] = new_df['Sector'].str.strip()

# 3. Apply the same transformations
new_df['Sector_enc'] = le.transform(new_df['Sector'])
X_new = new_df[['Profit 2022', 'Profit 2024', 'Sector_enc', 'Customer_Focused']]

# 4. Predict
preds = model.predict(X_new)

# 5. Output results
output = new_df[['Company Name']].copy()
output['WillBenefit'] = preds  # 1 = likely to profit, 0 = not
output.to_csv('../outputs/predictions.csv', index=False)
print('Saved predictions to outputs/predictions.csv')
