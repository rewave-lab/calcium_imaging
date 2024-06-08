import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sampling_rate = 20
baseline_time = 6 * sampling_rate  # seconds * Hz

# Calculate single neuron activity correlation for the US stimulus
sna_correlation = []

# Load z-score data for US stimulus
with open(r"demo/US-zscore-trial-by-trial.txt", "rb") as fp:
    act = pickle.load(fp)

# Consider activity after the baseline period
act = act[:, :, baseline_time:]

# Loop through each neuron
for i in range(act.shape[0]):
    aux_i = []
    # Loop through each trial for the current neuron
    for j in range(act.shape[1]):
        aux_j = []
        # Calculate correlation between the current trial and all other trials for the neuron
        for k in range(act.shape[1]):
            aux_j.append(np.corrcoef(act[i, j], act[i, k])[0, 1])
        aux_i.append(aux_j)
    aux_i = np.array(aux_i)
    
    # Calculate the mean correlation above the diagonal of the correlation matrix
    matrix = aux_i
    rows, cols = matrix.shape
    sum_above_diagonal = np.nansum(np.triu(matrix, k=1))
    count_above_diagonal = (cols * (cols - 1)) / 2  # Number of elements above the diagonal
    mean_above_diagonal = sum_above_diagonal / count_above_diagonal
    sna_correlation.append(mean_above_diagonal)

# Plot the distribution of single neuron activity correlations
bins = np.arange(-0.3, 1.2, 0.1)
plt.figure(figsize=(7, 5))
plt.hist(sna_correlation, bins=bins, density=True)
plt.ylabel("Density")
plt.xlabel("Single neuron activity correlation")
plt.title("US")
plt.tight_layout()
plt.show()

print("DONE!")
