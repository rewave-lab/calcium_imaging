import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bisect import bisect_left

# Setting random seed for reproducibility
np.random.seed(42)

cells = []

file = r"demo/Animal16-2023-04-23-101551.txt"

# Opening the original behavior file in read mode
with open(file, 'r') as fp:
    # Read and store all lines into a list
    lines = fp.readlines()
    # This reads all lines from the file and stores them in the 'lines' list.

# Writing to a new file while skipping the first 10 lines
with open(r"demo/behavior.csv", 'w') as fp:
    # Iterate through each line in the 'lines' list
    for number, line in enumerate(lines):
        # Check if the line number is not in the range of the first 10 lines (0 to 9 inclusive)
        if number not in np.arange(0, 10, 1, dtype=int):
            # Write the line to the new file
            fp.write(line)
            # This writes each line to the new file, excluding the first 10 lines.

# Function to read behavior data with consistent number of fields per line
def read_behavior_data(file_path):
    data = []
    # Open the file and iterate through each line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by spaces
            line_data = line.split()
            # Check if the line has the expected number of fields (3)
            if len(line_data) == 3:
                # If yes, append it to the data list
                data.append(line_data)
            else:
                # If not, skip the line or handle it according to your requirements
                pass
    # Create a DataFrame from the processed data
    df = pd.DataFrame(data, columns=['information_state', 'time', 'code'])
    return df

# Read behavior data using the function
df = read_behavior_data("demo/behavior.csv")

# Extracting events times
cue_times = np.array(df.loc[df["code"] == "3"]["time"], dtype=int)
reward_times = np.array(df.loc[df['code'] == "1"]["time"], dtype=int)
pokes = np.array(df.loc[df["code"] == "6"]["time"], dtype=int)
frames = np.array(df.loc[df["code"] == "11"]["time"], dtype=int)


# Reading traces of cells after manual selection
# Opening the file containing the traces data
with open("demo/MC.tiff_DF_filtered_all.txt", "rb") as fp:
    # Loading the traces data using pickle
    traces = pickle.load(fp)

# Printing the shape of the loaded traces data and the number of frames
print(traces.shape, len(frames))

sampling_rate = 20
baseline_time = 6 * sampling_rate  # seconds * Hz
end_windows = 3 * sampling_rate
reward_time = 7 * sampling_rate
exm_time_sd = 3 * sampling_rate

# Separating the trials of consumption
events = []
dist_reward = []
trial_true = []
for i in range(len(reward_times)):
    a = pokes[(reward_times[i] < pokes) & (pokes < reward_times[i] + 10000)]  # limit of 10s
    if len(a) > 0:
        events.append(a[0])
        trial_true.append(i)
        dist_reward.append((a[0] - reward_times[i]))

trial_true = np.array(trial_true)

# Checking consistency
if len(frames) != len(traces[0]):
    print(len(frames), len(traces[0]))
    print("ERROR between behavior and traces!")

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

for i in range(len(traces)):  # loop over cell ids
    aux2 = []
    for j in range(len(cues_consumption_frames)):
        aux2.append(traces[i][(cues_consumption_frames[j] - baseline_time):(consumption_frames[j] + end_windows + 1)])  # 20 frames per second
    cells.append(aux2)

# CUE z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][:9 * sampling_rate + 1])  # 6 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

# Shuffle data
for i in range(len(cells)):
    for j in range(len(cells[i])):
        for k in range(100):
            np.random.shuffle(cells_crop[i][j])

zscores = []
for i in range(len(cells_crop)):
    aux = []
    cell_baseline_std = np.std(np.array(cells_crop[i][:, :baseline_time]))
   
    for j in range(len(cells_crop[i])):
        avg = np.mean(cells_crop[i][j][:baseline_time])
        sd = np.std(cells_crop[i][j][:baseline_time])
        
        if sd < 0.05:  # detect only spurious activity
            sd = cell_baseline_std
        
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(r"demo/CS-zscore-trial-by-trial-shuffle.txt", "wb") as fp:
    pickle.dump(zscores, fp)
    
# CONSUMPTION z-scores
cells_crop = []
for i in range(len(cells)):
    aux_crop = []
    for j in range(len(cells[i])):
        aux_crop.append(cells[i][j][-9 * sampling_rate - 1:])  # 6 seconds before and 3 seconds after
    cells_crop.append(aux_crop)
cells_crop = np.array(cells_crop)

# Shuffle data
for i in range(len(cells)):
    for j in range(len(cells[i])):
        for k in range(100):
            np.random.shuffle(cells_crop[i][j])

zscores = []
for i in range(len(cells_crop)):
    aux = []
    cell_baseline_std = np.std(np.array(cells_crop[i][:, :baseline_time]))
   
    for j in range(len(cells_crop[i])):
        avg = np.mean(cells_crop[i][j][:baseline_time])
        sd = np.std(cells_crop[i][j][:baseline_time])
        
        if sd < 0.05:  # detect only spurious activity
            sd = cell_baseline_std
        
        aux.append((cells_crop[i][j] - avg) / sd)
    zscores.append(aux)
zscores = np.array(zscores)

with open(r"demo/US-zscore-trial-by-trial-shuffle.txt", "wb") as fp:
    pickle.dump(zscores, fp)
    
print("DONE!")

