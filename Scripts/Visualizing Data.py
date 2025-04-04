import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

df = pd.read_csv('/Users/c4ndlemandle/Downloads/final_dataset_cleaned.csv')
df = df.dropna(subset=['Profit 2022', 'Profit 2024'])
df['Profit Change'] = df['Profit 2024'] - df['Profit 2022']

lower = df['Profit Change'].quantile(0.01)
upper = df['Profit Change'].quantile(0.99)
df_filtered = df[(df['Profit Change'] >= lower) & (df['Profit Change'] <= upper)].sort_values(by='Profit Change')

profit_up = (df_filtered['Profit Change'] > 0).sum()
profit_down = (df_filtered['Profit Change'] < 0).sum()
print(f"📈 Profit Increase: {profit_up} companies")
print(f"📉 Profit Decrease: {profit_down} companies")

colors = df_filtered['Profit Change'].apply(lambda x: '#228B22' if x >= 0 else '#B22222')

sns.set_style("white")
fig, ax = plt.subplots(figsize=(14, 10))
bars = ax.bar(range(len(df_filtered)), df_filtered['Profit Change'], color=colors, width=1)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_title('Profit Change per Company', fontsize=18, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Profit Change ($)', fontsize=12)
ax.set_xticks([])
sns.despine(left=True, bottom=True)

ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

ax.text(10, df_filtered['Profit Change'].max() * 0.9, f"↑ {profit_up} companies", color='#228B22', fontsize=12)
ax.text(10, df_filtered['Profit Change'].min() * 0.9, f"↓ {profit_down} companies", color='#B22222', fontsize=12)

plt.tight_layout()
plt.show()
