from sklearn.cluster import KMeans
k = [2, 3, 4, 5, 6, 7, 8]
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

wss = []
for i in k:
    kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=47)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)
plt.plot(k, wss)
plt.xlabel("Value for k")
plt.ylabel("WSS")
plt.show()