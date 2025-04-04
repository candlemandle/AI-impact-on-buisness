import pandas as pd

file_path = '/Users/c4ndlemandle/Downloads/final_dataset.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df['Profit 2022'] = pd.to_numeric(df['Profit 2022'], errors='coerce')
df['Profit 2024'] = pd.to_numeric(df['Profit 2024'], errors='coerce')

df_clean = df.dropna(subset=['Profit 2022', 'Profit 2024'], how='all')
df_clean['Profit Change'] = df_clean['Profit 2024'] - df_clean['Profit 2022']

df_clean.to_csv('/Users/c4ndlemandle/Downloads/final_dataset_cleaned.csv', index=False)
print(f"✅ Cleaned dataset saved to: /Users/c4ndlemandle/Downloads/final_dataset_cleaned.csv")
