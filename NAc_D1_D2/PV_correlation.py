import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from itertools import product, combinations

warnings.simplefilter("ignore")

# Function to calculate population vector correlation
def pv_corr(i, j):
    est_i = i
    est_j = j
    aux_cor_same_bin = []
    for k in range(est_i.shape[1]):
        aux_cor_same_bin.append(np.corrcoef(est_i[:, k], est_j[:, k])[0, 1])
    return np.mean(aux_cor_same_bin)

data = []
days = [1, 5, 10]
target = "CS"

for i in range(len(days)):
    # Load z-score data for tracked neurons from different days
    with open(r"demo/population_vector_correlation/{}_animal_16_day{}.txt".format(target, days[i]), "rb") as fp:
        act1 = pickle.load(fp)

    aux_block = act1 
    int_half_block = int(aux_block.shape[0] / 2)

    block1 = np.mean(aux_block[:int_half_block, :, :], axis=0)
    block2 = np.mean(aux_block[int_half_block:, :, :], axis=0)

    data.append([block1, block2])

data = np.array(data)
day_trial = list(product(np.arange(data.shape[0]), np.arange(data.shape[1])))
# The first index refers to the day session index, and the second to the block of activity within the day session.

# All possible combinations of two blocks of activity across the three sessions
combinations = list(combinations(day_trial, 2))

within_session = []
between_sessions = []

for m in range(len(combinations)):
    i, j = combinations[m][0]
    k, l = combinations[m][1]

    if i == k:
        within_session.append(pv_corr(data[i, j], data[k, l]))
    else:
        between_sessions.append(pv_corr(data[i, j], data[k, l]))

print(target)
print("avg_within_session", np.mean(within_session))
print("avg_between_sessions", np.mean(between_sessions))

print("DONE!")

