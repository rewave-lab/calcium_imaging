import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings

# Ignore warnings
warnings.simplefilter("ignore")

# Set random seed for reproducibility
np.random.seed(42)
sampling_rate = 20
baseline_time = 6 * sampling_rate  # seconds * Hz

# Initialize lists to store accuracies
total_accuracy = []
CS_accuracy = []
US_accuracy = []
shuffle_total_accuracy = []
shuffle_CS_accuracy = []
shuffle_US_accuracy = []

# Auxiliary lists for accuracies
aux_1 = []
aux_2 = []
aux_3 = []
aux_4 = []
aux_5 = []
aux_6 = []

# Number of iterations
n_iterations = 1000

# Function to train and evaluate the SVM model
def train_and_evaluate(CS, US, aux_acc1, aux_acc2, aux_acc3, kk):
    CS_original = np.copy(CS)
    US_original = np.copy(US)
    
    for nn in range(CS_original.shape[1]):
        CS = CS_original[:, nn]
        US = US_original[:, nn]

        # Label the data
        y_CS = np.ones(len(CS)) * -10
        y_US = np.ones(len(US)) * 10
        y_all = np.concatenate([y_CS, y_US])

        # Split the data into training and validation sets
        X_train1, X_val1, y_train1, y_val1 = train_test_split(CS, y_CS, test_size=0.3, random_state=kk)
        X_train2, X_val2, y_train2, y_val2 = train_test_split(US, y_US, test_size=0.3, random_state=kk)

        X_train = np.concatenate([X_train1, X_train2])
        X_train = np.array([X_train]).T

        X_val = np.concatenate([X_val1, X_val2])
        X_val = np.array([X_val]).T

        X_train1 = np.array([X_train1]).T
        X_train2 = np.array([X_train2]).T
        X_val1 = np.array([X_val1]).T
        X_val2 = np.array([X_val2]).T

        y_train = np.concatenate([y_train1, y_train2])
        y_val = np.concatenate([y_val1, y_val2])

        # Initialize and train the SVM model
        svc_model = LinearSVC(C=0.8)
        svc_model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        predictions = svc_model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        aux_acc1.append(accuracy)

        predictions = svc_model.predict(X_val1)
        accuracy = accuracy_score(y_val1, predictions)
        aux_acc2.append(accuracy)

        predictions = svc_model.predict(X_val2)
        accuracy = accuracy_score(y_val2, predictions)
        aux_acc3.append(accuracy)

# Train and evaluate on actual data
for kk in range(n_iterations):
    if (kk%100==0):
        print(kk)
    with open(r"demo/CS-zscore-trial-by-trial.txt", "rb") as fp:
        CS = pickle.load(fp)
    CS = np.mean(CS[:, :, baseline_time:], axis=2).T

    with open(r"demo/US-zscore-trial-by-trial.txt", "rb") as fp:
        US = pickle.load(fp)
    US = np.mean(US[:, :, baseline_time:], axis=2).T
    
    CS_original = np.copy(CS)
    US_original = np.copy(US)

    train_and_evaluate(CS, US, aux_1, aux_2, aux_3, kk)

# Calculate mean accuracies for actual data
aux_1 = np.array(aux_1)
aux_2 = np.array(aux_2)
aux_3 = np.array(aux_3)

aux_1 = np.mean(aux_1.reshape((n_iterations, CS_original.shape[1])), axis=0)
aux_2 = np.mean(aux_2.reshape((n_iterations, CS_original.shape[1])), axis=0)
aux_3 = np.mean(aux_3.reshape((n_iterations, CS_original.shape[1])), axis=0)

total_accuracy.append(aux_1)
CS_accuracy.append(aux_2)
US_accuracy.append(aux_3)

# Train and evaluate on shuffled data
for kk in range(n_iterations):
    if (kk%100==0):
        print(kk)
    with open(r"demo/CS-zscore-trial-by-trial-shuffle.txt", "rb") as fp:
        CS = pickle.load(fp)
    CS = np.mean(CS[:, :, baseline_time:], axis=2).T

    with open(r"demo/US-zscore-trial-by-trial-shuffle.txt", "rb") as fp:
        US = pickle.load(fp)
    US = np.mean(US[:, :, baseline_time:], axis=2).T
    
    CS_original = np.copy(CS)
    US_original = np.copy(US)

    train_and_evaluate(CS, US, aux_4, aux_5, aux_6, kk)

# Calculate mean accuracies for shuffled data
aux_4 = np.array(aux_4)
aux_5 = np.array(aux_5)
aux_6 = np.array(aux_6)

aux_4 = np.mean(aux_4.reshape((n_iterations, CS_original.shape[1])), axis=0)
aux_5 = np.mean(aux_5.reshape((n_iterations, CS_original.shape[1])), axis=0)
aux_6 = np.mean(aux_6.reshape((n_iterations, CS_original.shape[1])), axis=0)

shuffle_total_accuracy.append(aux_4)
shuffle_CS_accuracy.append(aux_5)
shuffle_US_accuracy.append(aux_6)

# Plot the results
plt.boxplot([total_accuracy[0], shuffle_total_accuracy[0]])
for i, d in enumerate([total_accuracy[0], shuffle_total_accuracy[0]]):
    y = d
    x = np.random.normal(i + 1, 0.01, len(y))
    plt.scatter(x, y)
    
plt.xticks([1,2], ["Actual","Shuffle"])
plt.ylabel("Single cell decoding accuracy")
plt.tight_layout()
plt.show()
print("DONE")

