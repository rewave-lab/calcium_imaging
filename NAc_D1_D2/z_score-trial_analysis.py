import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bisect import bisect_left

cells = []

file = r"demo/Animal16-2023-04-23-101551.txt"

# Reading behavior file and skipping the first 10 lines
with open(file, 'r') as fp:
    lines = fp.readlines()
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
    df = pd.DataFrame(data, columns=['infomation_state', 'time', 'code'])
    return df

# Read behavior data
df = read_behavior_data("demo/behavior.csv")

# Extracting cue, reward, poke, and frame times
cue_times = np.array(df.loc[df["code"] == "3"]["time"], dtype=int)
reward_times = np.array(df.loc[df['code'] == "1"]["time"], dtype=int)
pokes = np.array(df.loc[df["code"] == "6"]["time"], dtype=int)
frames = np.array(df.loc[df["code"] == "11"]["time"], dtype=int)

# Reading traces of cells
with open("demo/MC.tiff_DF_filtered_all.txt", "rb") as fp:
    traces = pickle.load(fp)

print(traces.shape, len(frames))

# Define constants for analysis
sampling_rate = 20
baseline_time = 6 * sampling_rate  # 6 seconds * sampling rate
end_windows = 3 * sampling_rate
reward_time = 7 * sampling_rate
exm_time_sd = 3 * sampling_rate

# Separating consumption trials
events = []
dist_reward = []
trial_true = []
for i in range(len(reward_times)):
    a = pokes[(reward_times[i] < pokes) & (pokes < reward_times[i] + 10000)]  # 10s limit
    if len(a) > 0:
        events.append(a[0])
        trial_true.append(i)
        dist_reward.append((a[0] - reward_times[i]))

trial_true = np.array(trial_true)

# Checking consistency between frames and traces
if len(frames) != len(traces[0]):
    print(len(times), len(traces[0]))
    print("Error: inconsistency between behavior and traces!")

frames_specific = []
for i in range(len(events)):
    aux = []
    pos = bisect_left(frames, events[i])
    aux.append(np.where(frames[pos] == frames)[0][0])
    pos = bisect_left(frames, cue_times[trial_true[i]])
    aux.append(np.where(frames[pos] == frames)[0][0])
    frames_specific.append(aux)

frames_specific = np.array(frames_specific)
consumption_frames = frames_specific[:, 0]
cues_consumption_frames = frames_specific[:, 1]

for i in range(len(traces)):
    aux2 = []
    for j in range(len(cues_consumption_frames)):
        aux2.append(traces[i][(cues_consumption_frames[j] - baseline_time):(consumption_frames[j] + end_windows + 1)])
    cells.append(aux2)

# Calculating CUE z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][:9 * sampling_rate + 1])  # 6 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

zscores = []
for i in range(len(cells_crop)):
    aux = []
    cell_baseline_std = np.std(np.array(cells_crop[i][:, :baseline_time]))
    for j in range(len(cells_crop[i])):
        avg = np.mean(cells_crop[i][j][:baseline_time])
        sd = np.std(cells_crop[i][j][:baseline_time])
        if sd < 0.05:  # Detecting only spurious activity
            sd = cell_baseline_std
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(r"demo/CS-zscore-trial-by-trial.txt", "wb") as fp:
    pickle.dump(zscores, fp)

# Calculating CONSUMPTION z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][-9 * sampling_rate - 1:])  # 6 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

zscores = []
for i in range(len(cells_crop)):
    aux = []
    cell_baseline_std = np.std(np.array(cells_crop[i][:, :baseline_time]))
    for j in range(len(cells_crop[i])):
        avg = np.mean(cells_crop[i][j][:baseline_time])
        sd = np.std(cells_crop[i][j][:baseline_time])
        if sd < 0.05:  # Detecting only spurious activity
            sd = cell_baseline_std
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(r"demo/US-zscore-trial-by-trial.txt", "wb") as fp:
    pickle.dump(zscores, fp)

print("DONE!")
