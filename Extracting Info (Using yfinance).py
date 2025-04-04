import yfinance as yf
import pandas as pd

companies = {
    '3M Company': 'MMM',
    'Abbott Laboratories': 'ABT',
    'Abercrombie & Fitch Co.': 'ANF',
}

results = []

for name, ticker in companies.items():
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

    results.append({
        'Company Name': name,
        'Ticker': ticker,
        'Profit 2022': profit_2022,
        'Profit 2024': profit_2024
    })

df = pd.DataFrame(results)
df.to_csv('profits_2022_2024.csv', index=False)
print(df)
