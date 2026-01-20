# COSM Assignment
This repository contains my assignment for Hypothesis Testing and Regression Analysis.

## Folder Structure
- **Hypothesis Testing** — t-test, z-test, and combined test scripts.
- **Regression Analysis** — Linear and Multiple Linear Regression with dataset and results.
- Each subfolder includes a `result/` directory for output files and plots.

## How to Run
- Run each `.py` file in its folder.
- Internet access is optional. 
- If internet access is unavailable, modify the following lines in  
	`MRL1.py`, `LR1.py`, `Z-test.py`, `T-test.py`, and `BothT-Z.py`:
		url = "https://api.slingacademy.com/v1/sample-data/files/student-scores.csv"
		df = pd.read_csv(url)
	to: 
		df = pd.read_csv("student-scores.csv")
- Ensure dataset file (student-scores.csv) is in the same folder as the code scripts.

## Observation

In this dataset, the sample size is large (n ≈ 2000). For large samples, the t-distribution closely approximates the standard normal distribution.

As a result, the calculated t-statistic and Z-statistic converge to nearly identical values, producing the same p-values.

Therefore, the similarity in results is expected and confirms the theoretical relationship between the Z-test and t-test for large samples.

