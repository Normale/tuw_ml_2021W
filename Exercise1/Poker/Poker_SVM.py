# -*- coding: utf-8 -*-
"""Poker_SVM.ipynb
"""

import pathlib
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# ---------------- DEFINE FUNCTIONS ----------------

# Train the SVM on the training data and predict using the test data
def predict_svm(x_te, x_tr, y_tr):
    every_predictions = []
    svc = svm.SVC()
    svc.fit(x_tr, y_tr)
    prediction = svc.predict(x_te)
    every_predictions.append(prediction)
    return every_predictions


# Check the accuracy of given predictions on the test set y_test
def check_accuracy(y_test, predictions):
    ground_truth = y_test.to_list()
    size = len(ground_truth)
    lst = []

    for predict in predictions:
        count = 0
        for i, j in enumerate(ground_truth):
            if predict[i] == ground_truth[i]:
                count += 1
        lst.append(count / size)
    return lst


# ---------------- PREPARE DATA ----------------
# Read the data
df = pd.read_csv("../Datasets/PokerDataSet.csv", encoding="ISO-8859-1")
# print(df.head())

# Sample 5% of the data!
df = df.sample(50000, random_state=35)

# 1-HOT-ENCODING
categorized_df = df.iloc[:, [0, 2, 4, 6, 8]]
cat_1hot = pd.get_dummies(categorized_df.astype(str))
# cat_1hot.head()

# Now, because the 1 (Ace) is the highest card in Poker, we should change our data accordingly:
to_change = df.iloc[:, [1, 3, 5, 7, 9]]
changed = to_change.replace(range(1, 14), [13] + list(range(1, 13)))
# Ace is put on top (1 -> 13), Everything else is lowered (7->6 and 2->1 etc.)

prepped = changed.join(cat_1hot).join(df.iloc[:, 10])  # After all column preprocessing
# prepped.head()

# Split into input and target variables
X = prepped.iloc[:, :-1]  # Remove the ID and Class columns
Y = prepped.iloc[:, -1]

# Scale data
x = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_x_scaled = pd.DataFrame(x_scaled)
# df_x_scaled = X # USE THIS IF NO SCALING!!
# print(df_x_scaled)ame(x_test_scaled)

# ---------------- APPLY MODEL & TEST EFFICIENCY ----------------

r = []
t = []
runtime = []
testSizeRange = list(range(10, 35, 5))

print("\nTRAINING USING SUPPORT VECTOR MACHINES")
for testSize in testSizeRange:
    X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=testSize / 100, random_state=35)

    # SUPPORT VECTOR MACHINES
    start = timeit.default_timer()
    all_predictions = predict_svm(X_test, X_train, Y_train)
    stop = timeit.default_timer()
    time = stop - start
    print('Time: ', time)
    results = check_accuracy(Y_test, all_predictions)
    print("\nTest size = ", testSize / 100)
    print(results)
    testSize = testSize / 100
    r.append(results)
    t.append(testSize)
    runtime.append(time)

best_testSize = t[r.index(np.max(r))]

fig = plt.figure()
plt.scatter(t, r)
fig.suptitle('SVM with different test sizes', fontsize=14)
plt.xlabel('Test size', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(r), "with testsize = ", best_testSize)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Test size')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(t, r, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('runtime', color=color)  # we already handled the x-label with ax1
ax2.plot(t, runtime, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

folds = int(math.floor(1/best_testSize))

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with linear kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

clf = svm.SVC(kernel='rbf', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with rbf kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

clf = svm.SVC(kernel='poly', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with polynomial kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))
