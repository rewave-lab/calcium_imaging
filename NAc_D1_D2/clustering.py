import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import scipy

warnings.filterwarnings('ignore')

# Load data
with open(r"demo/zscore_D1_day1.txt", "rb") as fp: 
    data = pickle.load(fp)

# Standardize the data
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(data)

# Perform PCA and plot explained variance
pca = PCA()
pca.fit(segmentation_std)

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker="o", linestyle="--")
plt.xlim(0, 15)
plt.axhline(0.8, ls="--")
plt.axvline(12, ls="--")
plt.xticks(np.arange(0, 18, 2))
plt.title("Explained variance by components")
plt.xlabel("#Components")
plt.ylabel("Cumulative")
plt.tight_layout()
plt.show()

print("Based on the plot, we chose a dimensionality reduction to 12 dimensions.")

# Apply PCA with 12 components
pca = PCA(n_components=12)
scores_pca = pca.fit_transform(segmentation_std)

# Calculate similarity and silhouette scores for different cluster numbers
silhouette_scores = []
similarity_scores = []

for i in range(2, 11):
    kmeans_pca = KMeans(n_clusters=i, init="k-means++", random_state = 42, n_init=100)
    kmeans_pca.fit(scores_pca)
    labels = kmeans_pca.labels_
    
    aux_similarity = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        mu1 = np.mean(data[idx], axis=0)
        aux_similarity.extend(cosine_similarity([data[idx[j]]], [mu1])[0][0] for j in range(len(idx)))
    
    similarity_scores.append(np.mean(aux_similarity))
    silhouette_scores.append(silhouette_score(scores_pca, labels))

# Plot similarity and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(range(2, 11), similarity_scores, "o")
axes[0].set_xlabel("#Clusters - KMeans")
axes[0].set_ylabel("Average cosine similarity")
axes[0].set_ylim(0, 1)
axes[0].axvline(3, ls="--")

axes[1].plot(range(2, 11), silhouette_scores, "o")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("#Clusters - KMeans")
axes[1].set_ylabel("Silhouette score")
axes[1].axvline(3, ls="--")

fig.suptitle('Metrics converge to suggest the optimal number of clusters is 3')
plt.tight_layout()
plt.show()

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=42, n_iter=5000, n_jobs=-1)
tsne_results = tsne.fit_transform(segmentation_std)

# KMeans clustering with the optimal number of clusters (3)
optimal_clusters = 3
kmeans_pca = KMeans(n_clusters=optimal_clusters, init="k-means++", random_state=42, n_init=100)
kmeans_pca.fit(scores_pca)
labels = kmeans_pca.labels_

silhouette_avg = silhouette_score(scores_pca, labels)

# Plot silhouette analysis and clustering results
sample_silhouette_values = silhouette_samples(scores_pca, labels)

plt.figure(figsize=(14, 10))
plt.suptitle(f"k={optimal_clusters}, silhouette_avg = {silhouette_avg:.2f}")

plt.subplot(221)

y_lower = 10
for i in range(optimal_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = len(ith_cluster_silhouette_values)
    y_upper = y_lower + size_cluster_i

    color = plt.cm.tab10(i)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))
    
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", ls="--")

plt.subplot(222)
for k in range(optimal_clusters):
    plt.plot(tsne_results[:,0][labels == k],tsne_results[:,1][labels == k], "o")
plt.xlabel("#t-SNE1")
plt.ylabel("#t-SNE2")
    
plt.subplot(223)
for k in range(optimal_clusters):
    plt.scatter(scores_pca[:, 0][labels == k], scores_pca[:, 1][labels == k], label=f"Cluster {k+1}")

plt.xlabel("PCA1")
plt.ylabel("PCA2")

plt.subplot(224)
for k in range(optimal_clusters):
    plt.scatter(scores_pca[:, 0][labels == k], scores_pca[:, 2][labels == k], label=f"Cluster {k+1}")

plt.xlabel("PCA1")
plt.ylabel("PCA3")

plt.tight_layout()
plt.show()

# Reassign labels for better presentation
new_labels = np.ones(len(labels)) * 10
new_labels[labels == 0] = 1
new_labels[labels == 1] = 2
new_labels[labels == 2] = 3
kmeans_pca.labels_ = new_labels

# Define color map
def color_bar(sample, maxx=1.5, minn=-0.5):
    dark_position = -minn / (-minn + maxx)
    c = ["tab:cyan", "black", "orangered"]
    v = [0, dark_position, 1.]
    return LinearSegmentedColormap.from_list('rg', list(zip(v, c)), N=256)

# Plot heatmaps and cluster averages
n_clusters = [1, 2, 3]
fig, axs = plt.subplots(2, len(n_clusters) * 2, figsize=(6 * len(n_clusters) * 2, 9), gridspec_kw={'hspace': 0.05, 'wspace': 0.25})
axs = axs.ravel()

for j, cluster_num in enumerate(n_clusters):
    idx = np.where(kmeans_pca.labels_ == cluster_num)[0]
    data_p = data[idx]
    data_p_z = np.mean(data_p[:, 3*20:6*20], axis=1)
    
    for k in range(2):
        ax = axs[2*j + k]
        sns.heatmap(data_p[np.argsort(data_p_z)[::-1]][:, k*121:(k+1)*121], ax=ax, cbar=False, cmap=color_bar(data_p), vmax=1.5, vmin=-0.5, xticklabels=False)
        ax.axvline(x=3*20, lw=2, color="white", ls="--")
        ax.set_title(f"Cluster {cluster_num}, {'CS' if k == 0 else 'US'}")
        ax.set_yticks(np.arange(0, data_p.shape[0], 50))
    
    for k in range(2):
        ax = axs[2*j + k + len(n_clusters)*2]
        mean_response = np.mean(data[idx][:, k*121:(k+1)*121], axis=0)
        sem_response = scipy.stats.sem(data[idx][:, k*121:(k+1)*121])
        time = np.arange(0, len(mean_response), 1) * 50 / 1000 - 3
        
        ax.plot(time, mean_response, color="blue")
        ax.fill_between(time, mean_response + sem_response, mean_response - sem_response, facecolor="blue", alpha=0.5)
        ax.axvline(x=0, ls="--", color="k")
        ax.axhline(y=0, color='k', lw=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean DF/F Z-score")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 1.5)


plt.show()

print("DONE!")
