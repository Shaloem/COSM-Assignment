# ===============================================================
# Enhanced One-Sample Hypothesis Testing Script
# Supports both T-Test and Z-Test
# ===============================================================

import pandas as pd
import numpy as np
import scipy.stats as stats

def one_sample_test(data, population_mean, population_std=None, alpha=0.05):
    """
    Performs a one-sample T-test (default) or Z-test (if population_std is provided).
    Parameters:
    data (pd.Series): Sample data
    population_mean (float): Hypothesized population mean
    population_std (float, optional): Known population standard deviation (for Z-test)
    alpha (float): Significance level (default = 0.05)
    """
    
    # Drop missing values
    data = data.dropna()
    sample_mean = data.mean()
    sample_std = data.std(ddof=1)
    n = len(data)

    print("==================================================")
    print("One-Sample Hypothesis Test Results")
    print("==================================================")
    print(f"Sample Size (n): {n}")
    print(f"Sample Mean: {sample_mean:.4f}")
    print(f"Sample Std Dev: {sample_std:.4f}")
    print(f"Hypothesized Population Mean: {population_mean}")
    print("--------------------------------------------------")
    
    # Normality Check (Shapiro-Wilk)
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print("Normality Test (Shapiro-Wilk)")
    print(f"Statistic: {shapiro_stat:.4f}, P-value: {shapiro_p:.4f}")

    if shapiro_p > alpha:
        print("Data appears approximately normal (fail to reject H0 of normality).")
    else:
        print("Warning: Data may not be normally distributed.")
    
    print("--------------------------------------------------")
    
    # Decide between T-test and Z-test
    if population_std is None:
        # T-Test
        test_type = "One-Sample T-Test"
        test_stat, p_value = stats.ttest_1samp(data, population_mean)
        df = n - 1
        critical_value = stats.t.ppf(1 - alpha/2, df)
        standard_error = sample_std / np.sqrt(n)
    else:
        # Z-Test
        test_type = "One-Sample Z-Test"
        standard_error = population_std / np.sqrt(n)
        test_stat = (sample_mean - population_mean) / standard_error
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        critical_value = stats.norm.ppf(1 - alpha/2)
    
    print(f"{test_type}")
    print(f"Test Statistic: {test_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    # Confidence Interval
    margin_error = critical_value * standard_error
    ci_lower = sample_mean - margin_error
    ci_upper = sample_mean + margin_error
    
    print(f"{int((1-alpha)*100)}% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    # Effect Size (Cohen's d)
    cohens_d = (sample_mean - population_mean) / sample_std
    print(f"Cohen's d (Effect Size): {cohens_d:.4f}")
    
    # Hypothesis Decision
    print("--------------------------------------------------")
    if p_value < alpha:
        print(f"Decision: Reject the null hypothesis at α = {alpha}")
    else:
        print(f"Decision: Fail to reject the null hypothesis at α = {alpha}")
    
    print("==================================================\n")

# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    
    # Load dataset
    url = "https://api.slingacademy.com/v1/sample-data/files/student-scores.csv"
    df = pd.read_csv(url)
    
    # Select math scores
    math_scores = df['math_score']
    
    # Example 1: T-Test (unknown population std)
    one_sample_test(math_scores, population_mean=75)
    
    # Example 2: Z-Test (if population std were known, e.g., 10)
    # one_sample_test(math_scores, population_mean=75, population_std=10)
