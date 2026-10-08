import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate sample 2D dataset for clustering
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# 2. Initialize and fit the K-Means clustering model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# 3. Get cluster centroids
centroids = kmeans.cluster_centers_
# 4. Plot the clusters and centroids
plt.figure(figsize=(7, 5))
plt.scatter(
    X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap="viridis", label="Data Points"
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    s=200,
    alpha=0.8,
    marker="X",
    label="Centroids",
)

plt.title("K-Means Clustering (Unsupervised Learning)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("Cluster centroids calculated by K-Means:")
print(centroids)
