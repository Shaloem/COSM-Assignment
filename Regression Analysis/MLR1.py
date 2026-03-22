# Multiple Linear Regression on Student Scores Dataset
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1.Loading dataset
url = "https://api.slingacademy.com/v1/sample-data/files/student-scores.csv" # student-scores.csv
df = pd.read_csv(url)

# 2.Select relevant columns for multiple regression
# Let's predict math_score based on multiple predictors:
# 'weekly_self_study_hours', 'absence_days', 'part_time_job'
df['part_time_job'] = df['part_time_job'].astype(int)  # Convert boolean to int (1 for True, 0 for False)

# Independent variables (predictors)
X = df[['weekly_self_study_hours', 'absence_days', 'part_time_job']]

# Dependent variable (target)
y = df['math_score']

# 3.Visualize relationships (scatter matrix for multiple predictors)
sns.pairplot(df[['weekly_self_study_hours', 'absence_days', 'part_time_job', 'math_score']])
plt.show()

# 4.Train/test split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
    )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5.Fit multiple linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model coefficients
print(f"Intercept: {lr.intercept_:.2f}")
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})
print(coeff_df)

# 6.Predictions and evaluation
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMSE = {mse:.2f}\nR² = {r2:.3f}\n")

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R² = {adjusted_r2:.3f}")

# 7.Actual vs Predicted plot
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.title("Actual vs Predicted Math Scores")
plt.xlabel("Actual Math Scores")
plt.ylabel("Predicted Math Scores")
plt.show()

# 8.Residual analysis
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.show()

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted Math Scores")
plt.xlabel("Predicted Math Scores")
plt.ylabel("Residuals")
plt.show()
