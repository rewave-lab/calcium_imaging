import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set plot parameters for consistent styling
plt.rcParams["figure.figsize"] = [10, 6]
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['figure.facecolor'] = 'white'

# Function to create a color bar for heatmap
def color_bar(sample, maxx=None, minn=None):
    if minn is None:
        minn = np.min(sample)
    if maxx is None:
        maxx = np.max(sample)
    
    dark_position = -minn / (-minn + maxx)
    
    colors = ["tab:blue", "w", "tab:red"]
    values = [0, dark_position, 1.]
    color_list = list(zip(values, colors))
    print(dark_position)
    
    return LinearSegmentedColormap.from_list('rg', color_list, N=256)

# Load z-score data for CS and US
with open(r"demo/CS-zscore.txt", "rb") as fp:
    CS = pickle.load(fp)
with open(r"demo/US-zscore.txt", "rb") as fp:
    US = pickle.load(fp)

# Average the z-scores across trials for each neuron
CS = np.mean(CS, axis=1)
US = np.mean(US, axis=1)

# Prepare data for sorting neurons by their mean z-score response to CS
aux_sort = []
for i in range(len(CS)):
    aux_sort.append(np.mean(CS[i][60:]))  # Mean z-score after baseline

# Plotting heatmaps for CS and US aligned to the same neurons sorted by CS response
plt.figure(figsize=(14, 7))

# Heatmap for CS
ax = plt.subplot(121)
sns.heatmap(CS[np.argsort(aux_sort)[::-1]], ax=ax, cbar=False, cmap=color_bar(CS, maxx=1.5, minn=-0.5), vmax=1.5, vmin=-0.5,
            cbar_kws={"extend": "both", 'label': 'z-score'})
plt.axvline(x=60, lw=0.75, color="white")
plt.xticks(np.arange(0, 121, 20), np.arange(-3, 4, 1), rotation=0)
plt.title(f"CS - Total: {len(CS)}")
plt.xlabel("Time (s)")
plt.ylabel("# Neurons")

# Heatmap for US
ax = plt.subplot(122)
sns.heatmap(US[np.argsort(aux_sort)[::-1]], ax=ax, cbar=False, cmap=color_bar(US, maxx=1.5, minn=-0.5), vmax=1.5, vmin=-0.5,
            cbar_kws={"extend": "both", 'label': 'z-score'})
plt.axvline(x=60, lw=0.75, color="white")
plt.xticks(np.arange(0, 121, 20), np.arange(-3, 4, 1), rotation=0)
plt.title(f"US - Total: {len(US)}")
plt.xlabel("Time (s)")
plt.ylabel("# Neurons")

plt.tight_layout()
plt.show()

print("DONE!")
