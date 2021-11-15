# -*- coding: utf-8 -*-
"""Purchase_SVM.ipynb
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



df = pd.read_csv("../Datasets/speeddating_1.csv", encoding="ISO-8859-1")
# print(df.head())
filter_col = [col for col in df if col.startswith('d_')]

cols = set(df.columns)
categorized_cols = ["gender"] + [col for col in df if col.startswith('d_')]
categorized_cols.remove('d_age')
categorized_df = df[categorized_cols]

cat_1hot = pd.get_dummies(categorized_df)
encoded = df[["samerace", "met", "match"]]
encoded = encoded.replace('?', 0)
encoded = encoded.apply(pd.to_numeric)
a = (0, 1)
encoded = encoded[encoded['met'].isin(a)]
# print(encoded['met'].value_counts())
encoded.iloc[5550:5560]
# print(encoded.iloc[2420:2430])
cat_1hot = cat_1hot.join(encoded)
df[df.isna().any(axis=1)]
cat_1hot = cat_1hot[~cat_1hot.isna().any(axis=1)] #remove all (8) rows containing nulls
Y = cat_1hot.iloc[:,-1]
X = cat_1hot.iloc[:,:-1]
df_x_scaled = X



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
plt.xlabel('Test sie', fontsize=14)
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

folds = int(math.floor(1 / best_testSize))

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with linear kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy Linear = ", np.max(scores))

clf = svm.SVC(kernel='rbf', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with rbf kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy rbf = ", np.max(scores))

clf = svm.SVC(kernel='poly', C=1, random_state=42)
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(range(folds), scores)
fig.suptitle('SVM cross validation with test size ' + str(best_testSize) + '% with polynomial kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy polynomial = ", np.max(scores))
