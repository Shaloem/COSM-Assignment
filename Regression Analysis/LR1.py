# Linear Regression on Student Scores Dataset
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1.Dataset Loading
url = "https://api.slingacademy.com/v1/sample-data/files/student-scores.csv" # student-scores.csv
df = pd.read_csv(url)

print("Dataset loaded successfully!\n")
print(df.head(), "\n")

# 2.Selection of relevant coloumns for Linear Regression
# Prediction of math_score based on weekly_self_study_hours
X = df[['weekly_self_study_hours']]
y = df['math_score']

# 3.Visualization of it
sns.scatterplot(data=df, x='weekly_self_study_hours', y='math_score', color='blue')
plt.title("Weekly Study Hours vs Math Score")
plt.xlabel("Weekly Self Study Hours")
plt.ylabel("Math Score")
plt.show()

# 4.Test & train split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5.linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(f"Intercept: {lr.intercept_:.2f}")
print(f"Coefficient: {lr.coef_[0]:.2f}")

# 6.Predictions and evaluation
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMSE = {mse:.2f}\nR² = {r2:.3f}\n")

# 7.Actual vs Predicted plot
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title("Actual vs Predicted Math Scores")
plt.xlabel("Weekly Self Study Hours")
plt.ylabel("Math Score")
plt.legend()
plt.show()

# 8.Residual analysis
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.show()

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted Scores")
plt.xlabel("Predicted Math Scores")
plt.ylabel("Residuals")
plt.show()