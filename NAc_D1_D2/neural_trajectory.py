import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sampling_rate = 20
baseline_time = 3 * sampling_rate  # seconds * Hz

# Initialize lists to store Euclidean distances
euclidean = []
euclidean_mean = []

# Colors for CS and US events
CS_color = (16/255, 127/255, 64/255)
US_color = (255/255, 117/255, 4/255)

# Time vector
time = np.arange(0, 3 + 1/20, 1/20)

# Load z-score data for CS and US events
with open(r"demo/CS-zscore.txt", "rb") as fp:
    CS = pickle.load(fp)
CS = CS[:, :, baseline_time:]

with open(r"demo/US-zscore.txt", "rb") as fp:
    US = pickle.load(fp)
US = US[:, :, baseline_time:]

# Compute average responses for CS and US events
avg_CS = np.mean(CS, axis=1)
avg_US = np.mean(US, axis=1)

# Combine average responses
data = np.concatenate([avg_CS, avg_US], axis=1)
df = pd.DataFrame(data)
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df)

# Perform PCA
pca = PCA()
pca.fit(df.T)
reduced = pca.transform(df.T)

# Apply Gaussian filter
from scipy.ndimage import gaussian_filter1d
reduced = gaussian_filter1d(reduced, sigma=2, axis=0)

# Compute Euclidean distances between CS and US responses
all_euclidean = []
for mm in range(len(reduced[:, 0][:61])):
    all_euclidean.append(np.linalg.norm(reduced[:, :2][:61][mm] - reduced[:, :2][61:][mm]))
euclidean.append(all_euclidean)

# Compute Euclidean distances for individual trials
euclidean_trials = []
data_trials = np.concatenate([CS, US], axis=2)
aux_mean = np.mean(data_trials, axis=1)
aux_reduced = pca.transform(aux_mean.T)
aux_reduced = gaussian_filter1d(aux_reduced, sigma=2, axis=0)
all_euclidean = []
for mm in range(len(reduced[:, 0][:61])):
    all_euclidean.append(np.linalg.norm(aux_reduced[:, :2][:61][mm] - aux_reduced[:, :2][61:][mm]))
euclidean_trials.append(all_euclidean)

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.plot3D(reduced[:, 0][:61], reduced[:, 1][:61], time, color=CS_color, lw=2)
ax.plot3D(reduced[:, 0][61:], reduced[:, 1][61:], time, color=US_color, lw=2)
ax.set_zlabel("Time (s)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Plot individual trial trajectories
for i in range(data_trials.shape[1]):
    aux_reduced = pca.transform(data_trials[:, i, :].T)
    aux_reduced = gaussian_filter1d(aux_reduced, sigma=2, axis=0)
    ax.plot3D(aux_reduced[:, 0][:61], aux_reduced[:, 1][:61], time, color=CS_color, ls="--", alpha=0.5)
    ax.plot3D(aux_reduced[:, 0][61:], aux_reduced[:, 1][61:], time, color=US_color, ls="--", alpha=0.5)

plt.tight_layout()
plt.show()

#  Plot Euclidean distances across time
plt.plot(time, euclidean_trials[0])
plt.xlabel("Time (s)")
plt.ylabel("Euclidean distance")
plt.tight_layout()
plt.show()

print("DONE!")
