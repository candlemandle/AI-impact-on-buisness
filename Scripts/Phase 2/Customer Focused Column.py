import pandas as pd

# Load cleaned dataset from Phase 1
df = pd.read_csv('../../Data/Phase 1/final_dataset_cleaned.csv')
df['AI Benefits'] = df['AI Benefits'].astype(str)

# Define keywords related to customer engagement
keywords = ['customer', 'support', 'recommendation', 'marketing', 'crm']

# Create a new boolean column
df['Customer Focused'] = df['AI Benefits'].str.lower().apply(
    lambda x: any(keyword in x for keyword in keywords)
)

# Save updated dataset into Phase 2
df.to_csv('../../Data/Phase 2/final_dataset_phase2.csv', index=False)

# Print a quick summary
print(df['Customer Focused'].value_counts())
