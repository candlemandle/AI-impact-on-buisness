import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Load dataset
df = pd.read_csv('../../Data/Phase 2/final_dataset_phase2.csv')

# Group by sector and get average profit
sector_avg = df.groupby('Sector')['Profit Change'].mean().reset_index()
top10 = sector_avg.sort_values(by='Profit Change', ascending=False).head(10)

# Plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
bars = sns.barplot(data=top10, y='Sector', x='Profit Change', palette='crest')

# Format x-axis
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Add value labels
for bar in bars.patches:
    value = int(bar.get_width())
    y = bar.get_y() + bar.get_height() / 2
    ax.text(value, y, f'{value:,}', va='center', ha='left', fontsize=9, color='black')

# Labels
ax.set_title('Top 10 Sectors by Average Profit Change', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Profit Change ($)')
ax.set_ylabel('Sector')

plt.tight_layout()
plt.show()
