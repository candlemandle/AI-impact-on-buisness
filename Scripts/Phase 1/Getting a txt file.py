import pandas as pd

file_path = '../../Data/Phase 1/AI_AmericanCompaniesProfits.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

company_names = df['Company Name'].dropna().unique()
output_path = '../../Data/Phase 1/all_companies.txt'

with open(output_path, 'w') as f:
    for name in company_names:
        f.write(name.strip() + '\n')

print(f"✅ Saved all company names to: {output_path}")
