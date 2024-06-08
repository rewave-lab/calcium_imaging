import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Set sampling rate and baseline time
sampling_rate = 20
baseline_time = 6 * sampling_rate  # seconds * Hz

# Function to calculate tuning curve correlation
def tuning_curve(act, i, j):
    est_i = act[i]
    est_j = act[j]
    aux_cor_same_bin = []
    for k in range(act.shape[1]):
        aux_cor_same_bin.append(np.corrcoef(est_i[k], est_j[k])[0, 1])
    return np.nanmedian(aux_cor_same_bin)

# Function to calculate and plot tuning curve correlation
def calculate_tuning_curve_correlation(target):
    with open(f"demo/{target}-zscore-trial-by-trial.txt", "rb") as fp:
        act = np.transpose(pickle.load(fp), (1, 0, 2))[:, :, baseline_time:]

    aux_i = []
    for i in range(act.shape[0]):
        aux_j = []
        for j in range(act.shape[0]):
            aux_j.append(tuning_curve(act, i, j))
        aux_i.append(aux_j)

    matrix_tuning_curve = np.array(aux_i)

    # Create heatmap with Seaborn
    sns.heatmap(matrix_tuning_curve, cmap="viridis")
    plt.title(f"Tuning Curve Correlation - {target}")
    plt.xlabel("#trial")
    plt.ylabel("#trial")
    plt.show()

    matrix = matrix_tuning_curve

    # Calculate mean of the values above the main diagonal
    rows, cols = matrix.shape
    sum_above_diagonal = np.sum(np.triu(matrix, k=1))
    count_above_diagonal = (cols * (cols - 1)) / 2  # Number of elements above the diagonal
    mean_above_diagonal = sum_above_diagonal / count_above_diagonal

    pv_corr_per_trials = []

    # Calculate pairwise correlations per trial
    for i in range(rows):
        diagonal = []
        for j in range(min(i + 1, cols)):
            diagonal.append(matrix[i - j, cols - 1 - j])
        pv_corr_per_trials.append(diagonal)
    del pv_corr_per_trials[-1]

    mean_pv_correlation_per_trial = []

    # Plot pairwise correlations and their means
    for i in range(len(pv_corr_per_trials)):
        plt.plot(np.ones(len(pv_corr_per_trials[-(i + 1)])) * (i + 1), pv_corr_per_trials[-(i + 1)], "o", color="gray")
        mean_pv_correlation_per_trial.append(np.mean(pv_corr_per_trials[-(i + 1)]))

    plt.plot(np.arange(1, len(pv_corr_per_trials) + 1), mean_pv_correlation_per_trial, lw=3)
    plt.ylabel(f"Tuning Curve Correlation - {target}")
    plt.xlabel("Distance Between Trials")
    plt.title(f"avg: {mean_above_diagonal:.2f}")
    plt.tight_layout()
    plt.ylim(-0.6, 1)
    plt.show()

# Calculate and plot for target "CS"
calculate_tuning_curve_correlation("CS")

# Calculate and plot for target "US"
calculate_tuning_curve_correlation("US")

print("DONE!")

