import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# 1. Generate noisy dataset
np.random.seed(42)
X = np.random.rand(30, 1) * 4 - 2
y = X**3 - 2 * X**2 + X + np.random.randn(30, 1) * 2

# 2. Expand to 10th-degree polynomial features (increases risk of overfitting)
poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.3, random_state=42
)

# 3. Fit standard Linear Regression (Overfits easily)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 4. Fit Ridge Regression (L2 Regularization)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# 5. Fit Lasso Regression (L1 Regularization)
lasso_reg = Lasso(alpha=0.1, max_iter=10000)
lasso_reg.fit(X_train, y_train)

# 6. Evaluate and compare Test MSE
print("=== Regularization vs Overfitting Results ===")
print(
    f"Standard Linear Regression Test MSE: {mean_squared_error(y_test, lin_reg.predict(X_test)):.4f}"
)
print(
    f"Ridge Regression (L2) Test MSE:      {mean_squared_error(y_test, ridge_reg.predict(X_test)):.4f}"
)
print(
    f"Lasso Regression (L1) Test MSE:      {mean_squared_error(y_test, lasso_reg.predict(X_test)):.4f}"
)

