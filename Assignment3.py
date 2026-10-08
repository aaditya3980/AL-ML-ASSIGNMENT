import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1. Load a high-dimensional dataset (Iris dataset has 4 features)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(f"Original shape: {X.shape} (4 features)")

# 2. Apply PCA to reduce from 4 dimensions to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Reduced shape: {X_pca.shape} (2 features)")
print(
    f"Explained Variance Ratio: {pca.explained_variance_ratio_}"
)  # How much information was retained
# 3. Visualize the 2D projected data
plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolors="k"
)
plt.title("PCA: Dimensionality Reduction (4D to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(
    handles=scatter.legend_elements()[0], labels=list(iris.target_names)
)
plt.show()
