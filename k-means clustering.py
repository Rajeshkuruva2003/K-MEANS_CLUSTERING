import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulate customer purchase data
np.random.seed(42)  # For reproducibility

# Number of customers
num_customers = 100

# Simulate features: Total purchases in dollars and frequency of purchases
total_purchases = np.random.randint(100, 5000, size=num_customers)  # Total purchase amount
purchase_frequency = np.random.randint(1, 50, size=num_customers)   # Number of purchases

# Combine features into a dataset
customer_data = np.column_stack((total_purchases, purchase_frequency))

# Apply K-Means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(customer_data)

# Cluster labels
labels = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_points = customer_data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')

plt.title("Customer Clustering Based on Purchase History")
plt.xlabel("Total Purchases ($)")
plt.ylabel("Purchase Frequency")
plt.legend()
plt.show()

# Print cluster assignments
for i in range(k):
    print(f"Cluster {i+1}:")
    print(customer_data[labels == i])
