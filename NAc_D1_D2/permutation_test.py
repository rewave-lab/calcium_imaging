import numpy as np
import matplotlib.pyplot as plt
import pickle
from bisect import bisect_left
import collections

# Function to format pie chart labels
def func(pct, allvalues):
    absolute = int(np.around(pct / 100. * np.sum(allvalues)))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# Permutation test to determine the significance of differences before and after a stimulus
def permutation_test(before, after, num_permutations):
    observed_diff = np.mean(after) - np.mean(before)
    concatenated = np.concatenate((before, after))
    perm_diffs = np.zeros(num_permutations)
    
    for i in range(num_permutations):
        np.random.shuffle(concatenated)
        perm_diffs[i] = np.mean(concatenated[:len(before)]) - np.mean(concatenated[len(before):])
    
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / num_permutations
    
    return p_value, observed_diff

# Load CS and US data
with open(r"demo/CS-DF_F.txt", "rb") as fp:
    CS = pickle.load(fp)
with open(r"demo/US-DF_F.txt", "rb") as fp:
    US = pickle.load(fp)

# Define constants for analysis
sampling_rate = 20
baseline_time = 3 * sampling_rate
end_windows = 3 * sampling_rate
reward_time = 7 * sampling_rate
exm_time_sd = 3 * sampling_rate

# Calculate mean activity for CS and US across trials
CS = np.mean(CS, axis=1)
US = np.mean(US, axis=1)

# Function to classify neuronal responses
def classify_responses(target_stimulus):
    classification = [] 
    for i in range(len(target_stimulus)):
        before = target_stimulus[i][:baseline_time]
        after = target_stimulus[i][baseline_time:]
        p, mean = permutation_test(before, after, 1000)
        
        if p > 0.05:
            classification.append(0)  # Nonresponsive
        else:
            classification.append(1 if mean > 0 else -1)  # Excited or Inhibited
    return np.array(classification)

# Classify responses to CS
np.random.seed(0)
classification = classify_responses(CS)

# Count the occurrences of each classification
outcome_list = collections.Counter(classification)
frequency = [outcome_list.get(-1, 0), outcome_list.get(0, 0), outcome_list.get(1, 0)]

# Plot pie chart for CS responses
plt.figure(figsize=(10, 6))
data = [frequency[2], frequency[0], frequency[1]]
patches, texts, pcts = plt.pie(data, labels=['Excited', 'Inhibited', 'Nonresponsive'], colors=["tab:red", "tab:cyan", "black"], autopct=lambda pct: func(pct, data))
plt.setp(pcts, color='white', fontweight='bold', fontsize=18)
plt.title(f"Response to: CS - Total: {len(classification)}")
plt.tight_layout()
plt.show()

# Save CS classification results
with open(r"demo/CS-activity_classification.txt", "wb") as fp:
    pickle.dump(classification, fp)

# Classify responses to US
classification = classify_responses(US)

# Count the occurrences of each classification
outcome_list = collections.Counter(classification)
frequency = [outcome_list.get(-1, 0), outcome_list.get(0, 0), outcome_list.get(1, 0)]

# Plot pie chart for US responses
plt.figure(figsize=(10, 6))
data = [frequency[2], frequency[0], frequency[1]]
patches, texts, pcts = plt.pie(data, labels=['Excited', 'Inhibited', 'Nonresponsive'], colors=["tab:red", "tab:cyan", "black"], autopct=lambda pct: func(pct, data))
plt.setp(pcts, color='white', fontweight='bold', fontsize=18)
plt.title(f"Response to: US - Total: {len(classification)}")
plt.tight_layout()
plt.show()

print("DONE!")

# Save US classification results
with open(r"demo/US-activity_classification.txt", "wb") as fp:
    pickle.dump(classification, fp)

