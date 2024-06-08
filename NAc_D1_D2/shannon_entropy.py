import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bisect import bisect_left
import math
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Setting random seed for reproducibility
np.random.seed(42)

# Set target for analysis
target = "CS"

# Load DF/F traces data
with open(f"demo/{target}-DF_F.txt", "rb") as fp:
    cells = pickle.load(fp)
    
# Load activity classification data
with open(f"demo/{target}-activity_classification.txt", "rb") as fp:
    classification = pickle.load(fp)
    
# Define constants
sampling_rate = 20
baseline_time = 3 * sampling_rate  # seconds * Hz

def permutation_test(before, after, num_permutations):
    """
    Perform a permutation test to compare means of before and after periods.
    """
    observed_diff = np.mean(after) - np.mean(before)
    concatenated = np.concatenate((before, after))
    perm_diffs = np.zeros(num_permutations)
    
    for i in range(num_permutations):
        np.random.shuffle(concatenated)
        perm_diffs[i] = np.mean(concatenated[:len(before)]) - np.mean(concatenated[len(before):])
    
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / num_permutations
    
    return p_value, observed_diff

def shannon_entropy(data):
    """
    Calculate Shannon entropy of the data.
    """
    counts = Counter(data)
    total_count = len(data)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

# Initialize lists to store results
percentage_response_cells = []
entropy = []
all_responses = []

# Analyze each cell
for j in range(len(cells)):
    response_aux = []
    for i in range(len(cells[j])):
        np.random.seed(0)
        before = cells[j][i][:baseline_time]
        after = cells[j][i][baseline_time:]

        p, mean_diff = permutation_test(before, after, 1000)

        if p > 0.05:
            response_aux.append(0)
        else:
            response_aux.append(1 if mean_diff > 0 else -1)
                
    response_aux = np.array(response_aux)
    all_responses.append(response_aux)
    entropy.append(shannon_entropy(response_aux))
    percentage_response_cells.append(np.sum(response_aux == classification[j]) / len(response_aux))
    
all_responses = np.array(all_responses)

# Plot histogram of Shannon entropy
plt.figure(figsize=(8,6))
ax = plt.subplot(111)
data = entropy
hist, bins = np.histogram(data, bins=np.arange(-0.1, 1.58 + 0.2, 0.1), density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:blue", alpha=1, lw=3)
df = pd.DataFrame({"x": bins[1:], "y": absolute_probabilities})

# Add vertical lines for specific entropy values
plt.axvline(np.mean(entropy), color="tab:orange", ls="--", lw=3)
plt.axvline(1.5, color="gray", lw=0.5)
plt.axvline(1.3709505944546687, color="gray", lw=0.5)
plt.axvline(1.1812908992306925, color="gray", lw=0.5)
plt.axvline(0.9219280948873623, color="gray", lw=0.5)
plt.axvline(0.5689955935892812, color="gray", lw=0.5)

# Add text annotations for percentages
plt.text(0.04, 0.9, '100%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.375, 0.9, '90%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.58, 0.9, '80%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.74, 0.9, '70%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.86, 0.9, '60%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.94, 0.9, '50%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.99, 0.9, '33%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

plt.axvline(0, color="gray", lw=0.5)
plt.axvline(1.585, color="gray", lw=0.5)

# Set plot limits and labels
plt.xlim(-0.05, 1.6)
plt.ylabel("Percentage")
plt.xlabel(f"Shannon's entropy - {target}")
plt.tight_layout()
plt.show()

# Plot heatmap of all responses
plt.figure(figsize=(6,8))
colors = ["tab:blue", "w", "tab:red"]
values = [0, 0.5, 1.0]
color_map = LinearSegmentedColormap.from_list('rg', list(zip(values, colors)), N=3)
ax_heatmap = sns.heatmap(all_responses, cmap=color_map, cbar=False)
plt.yticks(np.arange(0, all_responses.shape[0], 10) + 0.5, np.arange(0, all_responses.shape[0], 10), rotation=0)
plt.xticks(np.arange(0, all_responses.shape[1], 1) + 0.5, np.arange(1, all_responses.shape[1] + 1, 1), rotation=0)
ax_heatmap.set_facecolor('white')

# Add heatmap titles and labels
plt.title(target)
plt.ylabel("#neurons")
plt.xlabel("#trials")
plt.tight_layout()
plt.show()

# Convert the percentage of response cells to numpy array and multiply by 100
percentage_response_cells = np.array(percentage_response_cells) * 100

# Set up the figure size for the plot
plt.figure(figsize=(6, 5))

# Define bins for the histogram
bins = np.arange(-10, 115, 10)

# Plot data for inhibited neurons
data = percentage_response_cells[classification == -1]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:cyan", alpha=1, lw=2, ls="--")

# Plot data for excited neurons
data = percentage_response_cells[classification == 1]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:red", alpha=1, lw=2)

# Plot data for non-responsive neurons
data = percentage_response_cells[classification == 0]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="k", alpha=1, lw=2, ls="-.")

# Set labels and title for the plot
plt.xlabel('Percentage of Persisted Responses')
plt.ylabel('Probability')
plt.title(target)
plt.ylim(bottom=0)

# Add a vertical line at 70%
plt.axvline(70, ls="--", color="k")

# Adjust layout for better fit
plt.tight_layout()

# Display the plot
plt.show()


# Repeat the same analysis and visualization for the unconditioned stimulus (US)
target = "US"

# Load DF/F traces data
with open(f"demo/{target}-DF_F.txt", "rb") as fp:
    cells = pickle.load(fp)
    
# Load activity classification data
with open(f"demo/{target}-activity_classification.txt", "rb") as fp:
    classification = pickle.load(fp)

# Initialize lists to store results
percentage_response_cells = []
entropy = []
all_responses = []

# Analyze each cell
for j in range(len(cells)):
    response_aux = []
    for i in range(len(cells[j])):
        np.random.seed(0)
        before = cells[j][i][:baseline_time]
        after = cells[j][i][baseline_time:]

        p, mean_diff = permutation_test(before, after, 1000)

        if p > 0.05:
            response_aux.append(0)
        else:
            response_aux.append(1 if mean_diff > 0 else -1)
                
    response_aux = np.array(response_aux)
    all_responses.append(response_aux)
    entropy.append(shannon_entropy(response_aux))
    percentage_response_cells.append(np.sum(response_aux == classification[j]) / len(response_aux))
    
all_responses = np.array(all_responses)

# Plot histogram of Shannon entropy
plt.figure(figsize=(8,6))
ax = plt.subplot(111)
data = entropy
hist, bins = np.histogram(data, bins=np.arange(-0.1, 1.58 + 0.2, 0.1), density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:blue", alpha=1, lw=3)
df = pd.DataFrame({"x": bins[1:], "y": absolute_probabilities})

# Add vertical lines for specific entropy values
plt.axvline(np.mean(entropy), color="tab:orange", ls="--", lw=3)
plt.axvline(1.5, color="gray", lw=0.5)
plt.axvline(1.3709505944546687, color="gray", lw=0.5)
plt.axvline(1.1812908992306925, color="gray", lw=0.5)
plt.axvline(0.9219280948873623, color="gray", lw=0.5)
plt.axvline(0.5689955935892812, color="gray", lw=0.5)

# Add text annotations for percentages
plt.text(0.04, 0.9, '100%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.375, 0.9, '90%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.58, 0.9, '80%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.74, 0.9, '70%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.86, 0.9, '60%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.94, 0.9, '50%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.99, 0.9, '33%', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

plt.axvline(0, color="gray", lw=0.5)
plt.axvline(1.585, color="gray", lw=0.5)

# Set plot limits and labels
plt.xlim(-0.05, 1.6)
plt.ylabel("Percentage")
plt.xlabel(f"Shannon's entropy - {target}")
plt.tight_layout()
plt.show()

# Plot heatmap of all responses
plt.figure(figsize=(6,8))
colors = ["tab:blue", "w", "tab:red"]
values = [0, 0.5, 1.0]
color_map = LinearSegmentedColormap.from_list('rg', list(zip(values, colors)), N=3)
ax_heatmap = sns.heatmap(all_responses, cmap=color_map, cbar=False)
plt.yticks(np.arange(0, all_responses.shape[0], 10) + 0.5, np.arange(0, all_responses.shape[0], 10), rotation=0)
plt.xticks(np.arange(0, all_responses.shape[1], 1) + 0.5, np.arange(1, all_responses.shape[1] + 1, 1), rotation=0)
ax_heatmap.set_facecolor('white')

# Add heatmap titles and labels
plt.title(target)
plt.ylabel("#neurons")
plt.xlabel("#trials")
plt.tight_layout()
plt.show()

# Convert the percentage of response cells to numpy array and multiply by 100
percentage_response_cells = np.array(percentage_response_cells) * 100

# Set up the figure size for the plot
plt.figure(figsize=(6, 5))

# Define bins for the histogram
bins = np.arange(-10, 115, 10)

# Plot data for inhibited neurons
data = percentage_response_cells[classification == -1]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:cyan", alpha=1, lw=2, ls="--")

# Plot data for excited neurons
data = percentage_response_cells[classification == 1]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="tab:red", alpha=1, lw=2)

# Plot data for non-responsive neurons
data = percentage_response_cells[classification == 0]
hist, bins = np.histogram(data, bins=bins, density=True)
absolute_probabilities = hist * np.diff(bins) * 100
plt.plot(bins[1:], absolute_probabilities, drawstyle='steps', color="k", alpha=1, lw=2, ls="-.")

# Set labels and title for the plot
plt.xlabel('Percentage of Persisted Responses')
plt.ylabel('Probability')
plt.title(target)
plt.ylim(bottom=0)

# Add a vertical line at 70%
plt.axvline(70, ls="--", color="k")

# Adjust layout for better fit
plt.tight_layout()

# Display the plot
plt.show()

print("DONE!")
