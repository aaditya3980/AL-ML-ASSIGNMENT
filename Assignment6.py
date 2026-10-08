import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Generate sample regression data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Output model metrics and parameters
print("=== Linear Regression Results ===")
print(f"Intercept (b): {model.intercept_[0]:.4f}")
print(f"Slope Coefficient (m): {model.coef_[0][0]:.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared Score (R2): {r2_score(y_test, y_pred):.4f}")

# 6. Plot the regression line
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="blue", alpha=0.6, label="Data Points")
plt.plot(
    X_test,
    y_pred,
    color="red",
    linewidth=2,
    label=f"Fit Line: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}",
)
plt.title("Linear Regression Fit")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.show()
