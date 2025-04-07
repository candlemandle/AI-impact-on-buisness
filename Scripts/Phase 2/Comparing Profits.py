import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Load dataset
df = pd.read_csv('../../Data/Phase 2/final_dataset_phase2.csv')

# Label companies as gain/loss
df['Profit Gained'] = df['Profit Change'] > 0

# Group and count
grouped = df.groupby(['Customer Focused', 'Profit Gained']).size().reset_index(name='Count')
grouped['Customer Focused'] = grouped['Customer Focused'].replace({True: 'Customer-Focused', False: 'Non-Customer-Focused'})
grouped['Profit Gained'] = grouped['Profit Gained'].replace({True: 'Gained', False: 'Lost'})

# Plot setup
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'Gained': '#228B22', 'Lost': '#B22222'}

bars = sns.barplot(data=grouped, x='Customer Focused', y='Count', hue='Profit Gained', palette=colors, ax=ax)

# Add value labels (skip zero)
for bar in bars.patches:
    height = bar.get_height()
    if height == 0:
        continue
    x = bar.get_x() + bar.get_width() / 2
    ax.annotate(f'{int(height)} ↑',
                xy=(x, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                color='black',
                fontweight='bold')

# Title + style
ax.set_title('Number of Companies Gaining or Losing Profit by AI Focus', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Companies')
ax.set_xlabel('')
ax.legend(title='Profit Gained', loc='upper right')
plt.tight_layout()
plt.show()
