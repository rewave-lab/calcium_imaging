import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bisect import bisect_left

# Constants
sampling_rate = 20
baseline_time = 3 * sampling_rate  # 3 seconds * sampling rate
end_windows = 3 * sampling_rate  # 3 seconds window
reward_time = 7 * sampling_rate
exm_time_sd = 3 * sampling_rate

# File paths
behavior_file_path = r"demo/Animal16-2023-04-23-101551.txt"
processed_behavior_file_path = r"demo/behavior.csv"
traces_file_path = "demo/MC.tiff_DF_filtered_all.txt"
cs_df_f_file_path = r"demo/CS-DF_F.txt"
cs_zscore_file_path = r"demo/CS-zscore.txt"
us_df_f_file_path = r"demo/US-DF_F.txt"
us_zscore_file_path = r"demo/US-zscore.txt"

# Remove the first 10 lines from the original behavior file and save to a new file
with open(behavior_file_path, 'r') as fp:
    lines = fp.readlines()
with open(processed_behavior_file_path, 'w') as fp:
    for number, line in enumerate(lines):
        if number >= 10:
            fp.write(line)

# Function to read behavior data ensuring each line has the same number of fields
def read_behavior_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line_data = line.split()
            if len(line_data) == 3:
                data.append(line_data)
    df = pd.DataFrame(data, columns=['information_state', 'time', 'code'])
    return df

# Read the processed behavior data
df = read_behavior_data(processed_behavior_file_path)

# Extracting various event times
cue_times = np.array(df.loc[df["code"] == "3"]["time"], dtype=int)
reward_times = np.array(df.loc[df['code'] == "1"]["time"], dtype=int)
pokes = np.array(df.loc[df["code"] == "6"]["time"], dtype=int)
frames = np.array(df.loc[df["code"] == "11"]["time"], dtype=int)

# Load traces data
with open(traces_file_path, "rb") as fp:
    traces = pickle.load(fp)

print(traces.shape, len(frames))

# Separate consumption trials
events = []
dist_reward = []
trial_true = []

for i in range(len(reward_times)):
    a = pokes[(reward_times[i] < pokes) & (pokes < reward_times[i] + 10000)]  # limit of 10s
    if len(a) > 0:
        events.append(a[0])
        trial_true.append(i)
        dist_reward.append(a[0] - reward_times[i])

trial_true = np.array(trial_true)

# Check consistency between frames and traces
if len(frames) != len(traces[0]):
    print(len(frames), len(traces[0]))
    print("ERROR: Mismatch between behavior frames and traces!")

# Identify specific frames for consumption and cues
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

# Crop and process traces for each cell
cells = []
for i in range(len(traces)):
    aux2 = []
    for j in range(len(cues_consumption_frames)):
        start = cues_consumption_frames[j] - baseline_time
        end = consumption_frames[j] + end_windows + 1
        aux2.append(traces[i][start:end])  # 20 frames per second
    cells.append(aux2)

# Calculate CUE z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][:6 * sampling_rate + 1])  # 3 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

with open(cs_df_f_file_path, "wb") as fp:
    pickle.dump(cells_crop, fp)

zscores = []
for i in range(len(cells_crop)):
    aux = []
    avg = np.mean(cells_crop[i][:, :baseline_time])
    sd = np.std(cells_crop[i][:, :baseline_time])
    for j in range(len(cells_crop[i])):
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(cs_zscore_file_path, "wb") as fp:
    pickle.dump(zscores, fp)

# Calculate CONSUMPTION z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][-6 * sampling_rate - 1:])  # 3 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

with open(us_df_f_file_path, "wb") as fp:
    pickle.dump(cells_crop, fp)

zscores = []
for i in range(len(cells_crop)):
    aux = []
    avg = np.mean(cells_crop[i][:, :baseline_time])
    sd = np.std(cells_crop[i][:, :baseline_time])
    for j in range(len(cells_crop[i])):
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(us_zscore_file_path, "wb") as fp:
    pickle.dump(zscores, fp)
    
print("DONE!")

