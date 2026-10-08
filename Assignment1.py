import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Create a simple dataset
# Independent variables: Hours_Studied, Attendance_Pct
# Dependent variable: Exam_Score
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Attendance_Pct': [60, 65, 70, 75, 80, 85, 90, 95],
    'Exam_Score': [50, 53, 60, 65, 72, 78, 85, 92]
}
df = pd.DataFrame(data)

# 2. Measure correlation matrix between all variables
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# 3. Visualize correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Heatmap (Dependent vs Independent Variables)")
plt.show()

# 4. Separate Independent (X) and Dependent (y) variables
X = df[['Hours_Studied', 'Attendance_Pct']]
y = df['Exam_Score']

# 5. Fit a Linear Regression model
model = LinearRegression()
model.fit(X, y)
for feature, coef in zip(X.columns, model.coef_):
    print(f"Association Strength (Coefficient) of {feature} with Exam_Score: {coef:.2f}")

 
