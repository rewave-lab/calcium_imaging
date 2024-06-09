import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import math

# Set sampling rate and baseline time
sampling_rate = 20
baseline_time = 3 * sampling_rate  # seconds * Hz

# Load CS data
with open(r"demo/CS-zscore.txt", "rb") as fp: 
    CS = pickle.load(fp)

# Average over all trials and then over the response time to CS
CS = np.mean(np.mean(CS, axis=1)[:, baseline_time:], axis=1) 

# Load US data
with open(r"demo/US-zscore.txt", "rb") as fp: 
    US = pickle.load(fp)

# Average over all trials and then over the response time to US
US = np.mean(np.mean(US, axis=1)[:, baseline_time:], axis=1) 

plt.figure(figsize=(10, 5))

# Scatter plot and linear regression
plt.subplot(121)
plt.axhline(0, color="k", zorder=0)
plt.axvline(0, color="k", zorder=0)

x = CS
y = US

angles = []
for i in range(len(x)):
    theta = math.atan2(y[i], x[i])
    angle_degrees = math.degrees(theta)
    angles.append(angle_degrees)

# Uncomment below code for larger datasets to remove outliers
# def find_outlier_indices_iqr(data):
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outlier_indices = [index for index, x in enumerate(data) if x < lower_bound or x > upper_bound]
#     return outlier_indices

# out_idx = []
# for i in [x, y]:
#     out_idx.append(find_outlier_indices_iqr(i))
# out_idx = np.unique(np.concatenate(out_idx).astype(int))

# x = np.delete(x, out_idx)
# y = np.delete(y, out_idx)
# angles = np.delete(angles, out_idx)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
angle = math.degrees(math.atan(slope))
correlation, p_value = stats.pearsonr(x, y)
print(stats.pearsonr(x, y))

# Calculate R-squared value
r_squared = r_value ** 2

# Linear function
def fun_lin(x, intercept, slope):
    return intercept + x * slope

# Plot data points and regression line
plt.plot(x, y, "ko")
plt.plot([np.min(x), np.max(x)], [fun_lin(np.min(x), intercept, slope), fun_lin(np.max(x), intercept, slope)], lw=3)

if p_value < 0.05:
    plt.title(f"r = {correlation:.2f}*, {angle:.2f}º")
else:
    plt.title(f"r = {correlation:.2f}, {angle:.2f}º")

plt.xlim(-1, 1.5)
plt.ylim(-1, 1)
plt.xlabel("CS response")
plt.ylabel("US response")

# Plot histogram of angle differences
plt.subplot(122)

angles = np.array(angles)
angles[angles < 0] = angles[angles < 0] + 360

hist, bin_edges = np.histogram(np.diff(angles[np.argsort(angles)]), bins=np.arange(0, 15, 0.5), density=True)

# Calculate midpoints (x) of bins
x = (bin_edges[:-1] + bin_edges[1:]) / 2

# y values (frequency)
y = hist

plt.hist(np.diff(angles[np.argsort(angles)]), bins=np.arange(0, 15, 0.5), density=True)
plt.ylabel("Density")
plt.xlabel("Angle between\nnearest neighbors")
plt.title(f"Avg. angle = {np.mean(np.diff(angles[np.argsort(angles)])):.1f}")
plt.ylim(0, 1.5)
plt.tight_layout()
plt.show()

print("DONE!")

