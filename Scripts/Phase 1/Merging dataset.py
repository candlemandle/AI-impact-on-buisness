import pandas as pd

original_path = '../../Data/Phase 1/AI_AmericanCompaniesProfits.csv'
ticker_path = '../../Data/Phase 1/company_tickers.csv'
output_path = '../../Data/Phase 1/enriched_dataset.csv'

original_df = pd.read_csv(original_path)
tickers_df = pd.read_csv(ticker_path)

original_df.columns = original_df.columns.str.strip()
tickers_df.columns = tickers_df.columns.str.strip()

merged_df = original_df.merge(tickers_df[['Company Name', 'Ticker']], on='Company Name', how='left')
merged_df.to_csv(output_path, index=False)
print("✅ Merged dataset with tickers saved.")
