# Z-Test Student Scores
# ===============================================================

import pandas as pd
import numpy as np
import scipy.stats as stats

# 1.Load dataset
url = "https://api.slingacademy.com/v1/sample-data/files/student-scores.csv"
df = pd.read_csv(url)

# 2.Select math scores for analysis
math_scores = df['math_score']

# 3.Z-Test (One Sample Z-Test)
# Assume population mean is 75 (as an example), and we don't know the population standard deviation
population_mean = 75
sample_mean = math_scores.mean()
sample_std = math_scores.std()
n = len(math_scores)

# Z-score formula: Z = (X̄ - μ) / (σ / sqrt(n))
z_score = (sample_mean - population_mean) / (sample_std / np.sqrt(n))

# Find the p-value (two-tailed)
z_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f" Z-Test Results:")
print(f"Sample Mean: {sample_mean:.2f}")
print(f"Z-score: {z_score:.2f}")
print(f"P-value: {z_p_value:.4f}\n")