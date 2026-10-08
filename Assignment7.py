import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# 1. Generate sample non-linear dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Sine wave with noise

# 2. Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Support Vector Regressor using an RBF (Radial Basis Function) kernel
svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# 4. Make predictions
y_pred = svr.predict(X_test)

# 5. Output performance metrics
print("=== Support Vector Regression (SVR) Results ===")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared Score (R2): {r2_score(y_test, y_pred):.4f}")

# 6. Plot SVR non-linear regression curve (Fixed using np.min and np.max)
X_grid = np.arange(np.min(X), np.max(X), 0.01)[:, np.newaxis]
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="darkorange", label="Data Points")
plt.plot(
    X_grid, svr.predict(X_grid), color="navy", lw=2, label="SVR RBF Model"
)
plt.title("Support Vector Regression (SVR)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
