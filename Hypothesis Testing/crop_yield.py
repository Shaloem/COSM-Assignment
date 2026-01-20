# 2 Sample Independent t- Test.
# Comparing 2 seperate grps (fert A & fert B)
# Testing whether the mean crop yield differs btw 2 independent groups (Fert A vs Fert B) under the same water condition.

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('crop_yield.csv')

def analyze_water_level(water_level):
    df_filtered = df[df['Water'] == water_level]
    yield_A = df_filtered[df_filtered['Fert'] == 'A']['Yield']
    yield_B = df_filtered[df_filtered['Fert'] == 'B']['Yield']
    t_stat, p_val = stats.ttest_ind(yield_A, yield_B)
    print(f"\n--- {water_level} Water ---")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Reject H₀: Fertilizer does affect yield.")
    else:
        print("Failed to reject H₀: No significant difference found.")

analyze_water_level("High")
analyze_water_level("Low")

# Combined plot
sns.boxplot(x='Water', y='Yield', hue='Fert', data=df)
plt.title('Crop Yield by Fertilizer Type Across Water Levels')
plt.xlabel('Water Level')
plt.ylabel('Yield')
plt.legend(title='Fertilizer')
plt.tight_layout()
plt.show()