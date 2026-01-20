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

# Assume population mean is 75 (as an example), and we don't know the population standard deviation
population_mean = 75
sample_mean = math_scores.mean()
sample_std = math_scores.std()
n = len(math_scores)

# 3.T-Test (One Sample T-Test)
# T-score formula: T = (X̄ - μ) / (S / sqrt(n))
t_score, t_p_value = stats.ttest_1samp(math_scores, population_mean)

print(f" T-Test Results:")
print(f"Sample Mean: {sample_mean:.2f}")
print(f"T-score: {t_score:.2f}")
print(f"P-value: {t_p_value:.4f}")
