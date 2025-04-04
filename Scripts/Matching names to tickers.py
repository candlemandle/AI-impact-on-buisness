import pandas as pd
from fuzzywuzzy import process

input_path = '/Users/c4ndlemandle/Downloads/all_companies.txt'
ticker_data_path = '/Users/c4ndlemandle/Downloads/companies.csv'
output_path = '/Users/c4ndlemandle/Downloads/company_tickers.csv'

ticker_df = pd.read_csv(ticker_data_path)

with open(input_path, 'r') as f:
    company_names = [line.strip() for line in f if line.strip()]

def match_ticker(name, choices, threshold=80):
    result = process.extractOne(name, choices)
    if isinstance(result, tuple) and len(result) == 2:
        match, score = result
        if score >= threshold:
            return match, score
    return None, None

results = []
choices = ticker_df['company name'].tolist()

for company in company_names:
    match, score = match_ticker(company, choices)
    if match:
        ticker = ticker_df.loc[ticker_df['company name'] == match, 'ticker'].values[0]
        results.append({'Company Name': company, 'Matched Name': match, 'Ticker': ticker, 'Score': score})
        print(f"✅ Matched: {company} → {match} ({ticker}) [Score: {score}]")
    else:
        results.append({'Company Name': company, 'Matched Name': None, 'Ticker': 'NOT FOUND', 'Score': 0})
        print(f"❌ No match found for: {company}")

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"\n✅ Matching complete! Results saved to: {output_path}")
