import pandas as pd
import yfinance as yf
import time
import os

enriched_path = '../../Data/Phase 1/enriched_dataset.csv'
partial_path = '../../Data/Phase 1/partial_results.csv'
final_path = '../../Data/Phase 1/final_dataset.csv'

enriched_df = pd.read_csv(enriched_path)
enriched_df.columns = enriched_df.columns.str.strip()

if os.path.exists(partial_path):
    partial_df = pd.read_csv(partial_path)
    processed_tickers = set(partial_df['Ticker'].dropna())
else:
    partial_df = pd.DataFrame(columns=['Ticker', 'Profit 2022', 'Profit 2024'])
    processed_tickers = set()

tickers_to_fetch = enriched_df['Ticker'].dropna().unique()
remaining_tickers = [ticker for ticker in tickers_to_fetch if ticker not in processed_tickers]

new_results = []

for count, ticker in enumerate(remaining_tickers, 1):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        available_columns = financials.columns.tolist()

        profit_2022 = None
        profit_2024 = None

        for col in available_columns:
            col_str = str(col)
            if '2022' in col_str:
                profit_2022 = financials.loc['Net Income', col]
            if '2023' in col_str or '2024' in col_str:
                profit_2024 = financials.loc['Net Income', col]

        new_results.append({'Ticker': ticker, 'Profit 2022': profit_2022, 'Profit 2024': profit_2024})
        print(f"{ticker}: 2022 → {profit_2022}, 2024 → {profit_2024}")

    except Exception as e:
        print(f"Error fetching for {ticker}: {e}")

    if count % 10 == 0 or count == len(remaining_tickers):
        if new_results:
            new_df = pd.DataFrame(new_results)
            partial_df = pd.concat([partial_df, new_df], ignore_index=True)
            partial_df.to_csv(partial_path, index=False)
            print(f"💾 Checkpoint saved after {count} tickers.")
            new_results = []

    time.sleep(0.5)

final_df = enriched_df.merge(partial_df, on='Ticker', how='left')
final_df.to_csv(final_path, index=False)
print("✅ Final dataset with profits saved.")
