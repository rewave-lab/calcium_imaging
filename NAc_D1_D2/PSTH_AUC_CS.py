import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bisect import bisect_left
import scipy
from sklearn.metrics import auc

cells = []

sampling_rate = 20  # Hz
baseline_time = 3 * sampling_rate  # seconds * Hz
end_window = 7 * sampling_rate  # 10 cue + 10 extras
reward_time = 7 * sampling_rate
exam_time_sd = 3 * sampling_rate

file = r"demo/Animal16-2023-04-23-101551.txt"

# Opening the original behavior file in read mode
with open(file, 'r') as fp:
    lines = fp.readlines()

# Writing to a new file while skipping the first 10 lines
with open(r"demo/behavior.csv", 'w') as fp:
    for number, line in enumerate(lines):
        if number not in np.arange(0, 10, 1, dtype=int):
            fp.write(line)

# Function to read behavior data with a consistent number of fields per line
def read_behavior_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line_data = line.split()
            if len(line_data) == 3:
                data.append(line_data)
    df = pd.DataFrame(data, columns=['information_state', 'time', 'code'])
    return df

# Read behavior data using the function
df = read_behavior_data("demo/behavior.csv")

# Extracting cue times
cue_times = np.array(df.loc[df["code"] == "3"]["time"], dtype=int)

# Extracting reward times
reward_times = np.array(df.loc[df['code'] == "1"]["time"], dtype=int)

# Extracting poke times
poke_times = np.array(df.loc[df["code"] == "6"]["time"], dtype=int)

# Extracting frame times
frame_times = np.array(df.loc[df["code"] == "11"]["time"], dtype=int)

# Reading traces of cells after manual selection
with open("demo/MC.tiff_DF_filtered_all.txt", "rb") as fp:
    traces = pickle.load(fp)

# Check consistency between frame times and traces
if len(frame_times) != len(traces[0]):
    print(len(frame_times), len(traces[0]))
    print("ERROR between behavior and traces!")

# Separate consumption trials
events = []
dist_reward = []
trial_true = []

for i in range(len(reward_times)):
    consumption_events = poke_times[(reward_times[i] < poke_times) & (poke_times < reward_times[i] + 10000)]  # 10s limit
    if len(consumption_events) > 0:
        events.append(consumption_events[0])
        trial_true.append(i)
        dist_reward.append(consumption_events[0] - reward_times[i])

trial_true = np.array(trial_true)

# Find specific frames related to events
frames_specific = []

for i in range(len(events)):
    aux = []
    pos = bisect_left(frame_times, events[i])
    aux.append(np.where(frame_times[pos] == frame_times)[0][0])

    pos = bisect_left(frame_times, cue_times[trial_true[i]])
    aux.append(np.where(frame_times[pos] == frame_times)[0][0])

    frames_specific.append(aux)

frames_specific = np.array(frames_specific)
consumption_frames = frames_specific[:, 0]
cue_frames = frames_specific[:, 1]

# Extract cell traces around cues and consumption events
for i in range(len(traces)):
    aux2 = []
    for j in range(len(cue_frames)):
        aux2.append(traces[i][(cue_frames[j] - baseline_time):(consumption_frames[j] + end_window + 1)])  # 20 frames per second
    cells.append(aux2)

# Calculate z-scores for CUE
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][:10 * sampling_rate + 1])  # 3 seconds before and 7 seconds after
    cells_crop.append(aux_crop)

cells_crop = np.array(cells_crop)

zscores = []
for i in range(len(cells_crop)):
    aux = []
    avg = np.mean(cells_crop[i][:, :baseline_time])
    sd = np.std(cells_crop[i][:, :baseline_time])
    for j in range(len(cells_crop[i])):
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)

zscores = np.array(zscores)

# Calculate average DF/F over all trials
avg_all_trials = []
for i in range(len(cells_crop)):
    avg_all_trials.append(np.mean(cells_crop[i], axis=0))

avg_all_trials = np.array(avg_all_trials)

# Permutation test function
def permutation_test(before, after, num_permutations):
    observed_diff = np.mean(after) - np.mean(before)
    concatenated = np.concatenate((before, after))
    perm_diffs = np.zeros(num_permutations)

    for i in range(num_permutations):
        np.random.shuffle(concatenated)
        perm_diffs[i] = np.mean(concatenated[:len(before)]) - np.mean(concatenated[len(before):])

    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / num_permutations

    return p_value, observed_diff

# Classify cells based on response
classification = []
for i in range(len(avg_all_trials)):
    before = avg_all_trials[i][:baseline_time]
    after = avg_all_trials[i][baseline_time:baseline_time + exam_time_sd]
    p, mean = permutation_test(before, after, 1000)

    if p > 0.05:
        classification.append(0)
    else:
        if mean > 0:
            classification.append(1)
        else:
            classification.append(-1)

classification = np.array(classification)

# Calculate average z-score over all trials
avg_zscore_all_trials = []
for i in range(len(cells_crop)):
    avg_zscore_all_trials.append(np.mean(zscores[i], axis=0))

avg_zscore_all_trials = np.array(avg_zscore_all_trials)

# Plot PSTH (Peri-Stimulus Time Histogram)
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(111)

plt.plot(np.mean(avg_zscore_all_trials[classification == -1], axis=0), label='Inhibited', color="tab:cyan")
mu1 = np.mean(avg_zscore_all_trials[classification == -1], axis=0)
sigma1 = scipy.stats.sem(avg_zscore_all_trials[classification == -1])
plt.fill_between(range(201), mu1 + sigma1, mu1 - sigma1, facecolor="tab:cyan", alpha=0.5)

plt.plot(np.mean(avg_zscore_all_trials[classification == 1], axis=0), label='Excited', color="tab:red")
mu1 = np.mean(avg_zscore_all_trials[classification == 1], axis=0)
sigma1 = scipy.stats.sem(avg_zscore_all_trials[classification == 1])
plt.fill_between(range(201), mu1 + sigma1, mu1 - sigma1, facecolor="tab:red", alpha=0.5)

plt.axvline(x=baseline_time, color="orange", lw=3, ls="--")

plt.xticks(np.arange(0, 201, 20), np.arange(-3, 8, 1), rotation=0)
plt.xlabel("Time (s)")
plt.ylabel("Average z-score")
plt.xlim(0, 201)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.title("PSTH")
plt.ylim(-0.5, 1.5)
plt.tight_layout()
plt.show()

# Calculate AUC (Area Under the Curve) for inhibited cells
auc_inhibited = []
target = avg_zscore_all_trials[classification == -1]
x = np.arange(-3, 7 + 0.05, 1 / 20)

initial_limit = 0
final_limit = 3

# Find the indices corresponding to the limits in the x array
initial_index = np.argmax(x >= initial_limit)
final_index = np.argmax(x >= final_limit)

x_segment = x[initial_index:final_index + 1]

for i in range(len(target)):
    y_segment = target[i, initial_index:final_index + 1]
    auc_segment = auc(x_segment, y_segment)
    auc_inhibited.append(auc_segment)

df = pd.DataFrame(auc_inhibited)
df.to_excel(r"demo/AUC-CS-inhibited.xlsx", index=False, header=False)

# Calculate AUC (Area Under the Curve) for excited cells
auc_excited = []
target = avg_zscore_all_trials[classification == 1]
x = np.arange(-3, 7 + 0.05, 1 / 20)

initial_limit = 0
final_limit = 3

# Find the indices corresponding to the limits in the x array
initial_index = np.argmax(x >= initial_limit)
final_index = np.argmax(x >= final_limit)

x_segment = x[initial_index:final_index + 1]

for i in range(len(target)):
    y_segment = target[i, initial_index:final_index + 1]
    auc_segment = auc(x_segment, y_segment)
    auc_excited.append(auc_segment)

df = pd.DataFrame(auc_excited)
df.to_excel(r"demo/AUC-CS-excited.xlsx", index=False, header=False)

print("DONE!")

