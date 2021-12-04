# -*- coding: utf-8 -*-
"""Purchase_RF.ipynb
"""

import pathlib
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------- DEFINE FUNCTIONS ----------------

def predict_random_forest(x_te, x_tr, y_tr, kList):
    every_prediction = []
    for k in tqdm(kList):
        rndf = RandomForestClassifier(n_estimators=k, criterion="entropy")
        rndf.fit(x_tr, y_tr)
        prediction = rndf.predict(x_te)
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
pathTest = "../Datasets/purchase600-100cls-15k.tes.csv"
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
testSizeRange = list(range(10, 35, 5))
KList = [1, 5, 10, 50, 100, 1000]

print("\nTRAINING USING RANDOM FOREST")
for testSize in testSizeRange:
    X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=testSize / 100, random_state=35)

    # RANDOM FORESTS
    start = timeit.default_timer()
    all_predictions = predict_random_forest(X_test, X_train, Y_train, KList)
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
bestK = KList[idxs[1]]

fig = df.plot(title='Training with different k and test sizes', ylabel='Accuracy', xlabel='K')
fig.legend(testSizeRange)
# fig.subtitle('SVM with different test sizes', fontsize=14)
# df.xlabel('Test sie', fontsize=14)
# df.ylabel('Accuracy', fontsize=14)
# print("Max accuracy = ", np.max(r), "with testsize = ", best_testSize)

fig = plt.figure()
plt.scatter(testSizeRange, runtime)
fig.suptitle('Runtime', fontsize=14)
plt.xlabel('Test size', fontsize=14)
plt.ylabel('Run time', fontsize=14)

BestTestSize = int(math.floor(100/testSizeRange[idxs[0]]))
folds = math.floor(100 / BestTestSize)
clf = RandomForestClassifier(n_estimators=bestK)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('RF cross validation with test size ' + str(BestTestSize), fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max CV accuracy = ", np.max(scores))
