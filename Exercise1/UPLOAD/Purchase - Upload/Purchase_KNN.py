# -*- coding: utf-8 -*-
"""Purchase_SVM.ipynb
"""

import pathlib
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


# ---------------- DEFINE FUNCTIONS ----------------


# Use the k-NN method (up to k_max) to predict the output variable on x_test, using the training data
def predict_knn(x_te, x_tr, y_tr, k_min=1, k_max=25):
    every_prediction = []
    for k in tqdm(range(k_min, k_max)):
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        # Fit the classifier to the data
        knn.fit(x_tr, y_tr)
        # Predict on x_test
        prediction = knn.predict(x_te)
        every_prediction.append(prediction)
    return every_prediction


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
df = pd.read_csv("../Datasets/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
# print(df.head())

# Split into input and target variables
X = df.iloc[:, 1:-1]  # Remove the ID and Class columns
Y = df.iloc[:, -1]

# Scale data
x = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_x_scaled = pd.DataFrame(x_scaled)
# print(df_x_scaled)

# Import test-data and scaling the data
pathTest = "Datasets/purchase600-100cls-15k.tes.csv"
dirPathTest = pathlib.Path(pathTest)
df_test = pd.read_csv(dirPathTest)

xTest = df.iloc[:, 1:-1]  # Remove the ID and Class columns
x_test = xTest.values  # returns a numpy array

x_test_scaled = min_max_scaler.fit_transform(x_test)
df_test_normalized = pd.DataFrame(x_test_scaled)
# print(df_test_normalized)

# ---------------- APPLY MODEL & TEST EFFICIENCY ----------------

r = []
t = []
runtime = []
kMin = 1
kMax = 25
testSizeRange = list(range(5, 17, 2))

print("\nTRAINING USING SVM")
for testSize in testSizeRange:
    X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=testSize / 100, random_state=35)

    # SUPPORT VECTOR MACHINES
    start = timeit.default_timer()
    all_predictions = predict_knn(X_test, X_train, Y_train)
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

bestOverallResult = np.max(r)
df = pd.DataFrame(data=r)
idxs = df.stack().index[np.argmax(df.values)]
print("The best indexes (testsize, k): ", idxs)

fig = df.plot(title='Training with different k and test sizes', ylabel='Accuracy', xlabel='K')
fig.legend(testSizeRange)

fig = plt.figure()
plt.scatter(testSizeRange, runtime)
fig.suptitle('Runtime', fontsize=14)
plt.xlabel('Test size', fontsize=14)
plt.ylabel('Run time', fontsize=14)


optTestSize = 100/testSizeRange[idxs[0]]
print("For test size = " + str(optTestSize))

# EUCLIDIAN
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='euclidean')
scores = cross_val_score(clf, df_x_scaled, Y, cv=optTestSize)
x = range(kMin, kMax)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(optTestSize) + '% with euclidean distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# MANHATTAN
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='manhattan')
scores = cross_val_score(clf, df_x_scaled, Y, cv=optTestSize)
x = range(kMin, kMax)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(optTestSize) + '% with manhattan distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# CHEBYSHEV
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='chebyshev')
scores = cross_val_score(clf, df_x_scaled, Y, cv=optTestSize)
x = range(kMin, kMax)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(optTestSize) + '% with chebyshev distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# MINKOWSKI
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='minkowski')
scores = cross_val_score(clf, df_x_scaled, Y, cv=optTestSize)
x = range(kMin, kMax)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(optTestSize) + '% with minkowski distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))
