import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import warnings

warnings.simplefilter("ignore")

# Define constants
sampling_rate = 20
baseline_time = 3 * sampling_rate  # seconds * Hz

# Load data for each day, excluding the baseline time
with open("demo/CS_correlation_between_days/avg-z-score-an_16-track-CS-day-1.txt", "rb") as fp:
    act_day1 = pickle.load(fp)[:, baseline_time:]

with open("demo/CS_correlation_between_days/avg-z-score-an_16-track-CS-day-5.txt", "rb") as fp:
    act_day5 = pickle.load(fp)[:, baseline_time:]

with open("demo/CS_correlation_between_days/avg-z-score-an_16-track-CS-day-10.txt", "rb") as fp:
    act_day10 = pickle.load(fp)[:, baseline_time:]

# Function to compute correlations between activities of different days
def compute_correlations(day1, day2):
    correlations = []
    p_values = []
    for i in range(len(day1)):
        correlation, p_value = stats.pearsonr(day1[i], day2[i])
        correlations.append(correlation)
        p_values.append(p_value)
    return correlations, p_values

# Compute correlations for day1 vs day5, day1 vs day10, and day5 vs day10
cors_1_5, p_cors_1_5 = compute_correlations(act_day1, act_day5)
cors_1_10, p_cors_1_10 = compute_correlations(act_day1, act_day10)
cors_5_10, p_cors_5_10 = compute_correlations(act_day5, act_day10)

target = "CS"
days = ["1-5", "1-10", "5-10"]
data = [cors_1_5, cors_1_10, cors_5_10]
bins = np.arange(-1.1, 1.2, 0.1)

# Plot correlation histograms
plt.figure(figsize=(10, 5))
for i in range(len(days)):
    dados = data[i]
    hist, bins = np.histogram(dados, bins=bins, density=True)
    probabilities = hist * np.diff(bins) * 100
    plt.plot(bins[1:], probabilities, drawstyle='steps', color=plt.cm.tab10(i), alpha=1, lw=3, label=f"Day {days[i]}")
    plt.axvline(np.mean(dados), lw=2, color=plt.cm.tab10(i), ls="--")

plt.ylabel("Percentage")
plt.xlabel("Correlation coefficient")
plt.legend(loc='upper right')
plt.xticks(np.arange(-1, 1.2, 0.5))
plt.title(target.upper())
plt.tight_layout()
plt.show()

print("DONE!")

