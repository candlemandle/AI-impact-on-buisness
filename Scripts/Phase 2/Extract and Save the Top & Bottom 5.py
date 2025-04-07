import pandas as pd

# Load dataset with 'Customer Focused' column
df = pd.read_csv('../../Data/Phase 2/final_dataset_phase2.csv')

# Sort by Profit Change
top5 = df.sort_values(by='Profit Change', ascending=False).head(5)
bottom5 = df.sort_values(by='Profit Change', ascending=True).head(5)

# Combine for export
case_studies = pd.concat([top5, bottom5])

# Save for manual inspection
case_studies[['Company Name', 'Sector', 'Profit Change', 'Customer Focused', 'AI Benefits']].to_csv(
    '../../Data/Phase 2/case_study_top_bottom_5.csv', index=False
)

print("✅ Top 5 and Bottom 5 companies extracted and saved.")
