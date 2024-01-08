import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

# Generate sample data (replace this with your own dataset)
data = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
X = data[0]

# Define a range of k values to test
k_values = range(1, 11)

# Calculate the sum of squared distances (inertia) for each k
inertia = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in k_values]

# Plot the elbow curve
plt.plot(k_values, inertia, marker='o')
plt.title('Distortion Score Elbow Method for K-means Clustering')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('distortion score')
plt.grid(True)
plt.show()
# Arthur: Thanh Linh Le