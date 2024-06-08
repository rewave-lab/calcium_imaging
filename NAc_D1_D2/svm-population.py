import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

np.random.seed(42)
sampling_rate = 20
baseline_time = 6 * sampling_rate  # seconds * Hz

# Initialize lists to store accuracy scores
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

# Loop over iterations
for kk in range(1000):
    # Load CS and US data
    with open(r"demo/CS-zscore-trial-by-trial.txt", "rb") as fp:
        CS = pickle.load(fp)
    CS = np.mean(CS[:, :, baseline_time:], axis=2).T

    with open(r"demo/US-zscore-trial-by-trial.txt", "rb") as fp:
        US = pickle.load(fp)
    US = np.mean(US[:, :, baseline_time:], axis=2).T

    # Label the data
    y_CS = np.ones(len(CS)) * -10
    y_US = np.ones(len(US)) * 10

    # Split the data into training and validation sets
    X_train1, X_val1, y_train1, y_val1 = train_test_split(CS, y_CS, test_size=0.3, random_state=kk)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(US, y_US, test_size=0.3, random_state=kk)

    X_train = np.concatenate([X_train1, X_train2])
    X_val = np.concatenate([X_val1, X_val2])
    y_train = np.concatenate([y_train1, y_train2])
    y_val = np.concatenate([y_val1, y_val2])

    # Initialize and train the SVM model
    svc_model = LinearSVC(C=0.8)
    svc_model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    predictions = svc_model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    aux_1.append(accuracy)

    predictions = svc_model.predict(X_val1)
    accuracy = accuracy_score(y_val1, predictions)
    aux_2.append(accuracy)

    predictions = svc_model.predict(X_val2)
    accuracy = accuracy_score(y_val2, predictions)
    aux_3.append(accuracy)

# Calculate mean accuracies for actual data
total_accuracy.append(np.mean(aux_1))
CS_accuracy.append(np.mean(aux_2))
US_accuracy.append(np.mean(aux_3))

# Loop over iterations for shuffled data
for kk in range(1000):
    with open(r"demo/CS-zscore-trial-by-trial-shuffle.txt", "rb") as fp:
        CS = pickle.load(fp)
    CS = np.mean(CS[:, :, baseline_time:], axis=2).T

    with open(r"demo/US-zscore-trial-by-trial-shuffle.txt", "rb") as fp:
        US = pickle.load(fp)
    US = np.mean(US[:, :, baseline_time:], axis=2).T

    y_CS = np.ones(len(CS)) * -10
    y_US = np.ones(len(US)) * 10

    X_train1, X_val1, y_train1, y_val1 = train_test_split(CS, y_CS, test_size=0.3, random_state=kk)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(US, y_US, test_size=0.3, random_state=kk)

    X_train = np.concatenate([X_train1, X_train2])
    X_val = np.concatenate([X_val1, X_val2])
    y_train = np.concatenate([y_train1, y_train2])
    y_val = np.concatenate([y_val1, y_val2])

    svc_model = LinearSVC(C=0.8)
    svc_model.fit(X_train, y_train)

    predictions = svc_model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    aux_4.append(accuracy)

    predictions = svc_model.predict(X_val1)
    accuracy = accuracy_score(y_val1, predictions)
    aux_5.append(accuracy)

    predictions = svc_model.predict(X_val2)
    accuracy = accuracy_score(y_val2, predictions)
    aux_6.append(accuracy)

# Calculate mean accuracies for shuffled data
shuffle_total_accuracy.append(np.mean(aux_4))
shuffle_CS_accuracy.append(np.mean(aux_5))
shuffle_US_accuracy.append(np.mean(aux_6))

print("Actual", total_accuracy)
print("Shuffle", shuffle_total_accuracy)
print("DONE!")
