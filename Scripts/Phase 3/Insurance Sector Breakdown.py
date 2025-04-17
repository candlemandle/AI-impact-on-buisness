import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Load data
df = pd.read_csv('../../Data/Phase 2/final_dataset_phase2.csv')

# Filter for Insurance companies
insurance_df = df[df['Sector'] == 'Insurance'].copy()

# Sort by Profit Change
insurance_df = insurance_df.sort_values(by='Profit Change', ascending=False)

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
bars = sns.barplot(data=insurance_df, y='Company Name', x='Profit Change', palette='crest')

# Format numbers
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Add labels
for bar in bars.patches:
    width = int(bar.get_width())
    y = bar.get_y() + bar.get_height() / 2
    plt.text(width, y, f'{width:,}', va='center', ha='left', fontsize=9)

# Titles
plt.title('Profit Change – Insurance Sector (Top Companies)', fontsize=14, fontweight='bold')
plt.xlabel('Profit Change ($)')
plt.ylabel('Company')

plt.tight_layout()
plt.show()
