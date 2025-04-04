import pandas as pd

file_path = '/Users/c4ndlemandle/Downloads/AI_AmericanCompaniesProfits.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

company_names = df['Company Name'].dropna().unique()
output_path = '/Users/c4ndlemandle/Downloads/all_companies.txt'

with open(output_path, 'w') as f:
    for name in company_names:
        f.write(name.strip() + '\n')

print(f"Saved company names to: {output_path}")
