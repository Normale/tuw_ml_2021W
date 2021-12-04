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
kMin = 1
kMax = 25
testSizeRange = list(range(5, 17, 2))

print("\nTRAINING USING KNN")
for testSize in testSizeRange:
    X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=testSize / 100, random_state=35)

    # KNN
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

best_testSize = testSizeRange[idxs[0]]
folds = int(math.floor(100/best_testSize))

fig = df.plot(title='Training with different k and test sizes', ylabel='Accuracy', xlabel='K')
fig.legend(testSizeRange)

fig = plt.figure()
plt.scatter(testSizeRange, runtime)
fig.suptitle('Runtime', fontsize=14)
plt.xlabel('Test size', fontsize=14)
plt.ylabel('Run time', fontsize=14)


print("For test size = " + str(best_testSize))

# EUCLIDIAN
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='euclidean')
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)
x = range(folds)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(best_testSize) + '% with euclidean distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# MANHATTAN
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='manhattan')
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(best_testSize) + '% with manhattan distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# CHEBYSHEV
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='chebyshev')
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(best_testSize) + '% with chebyshev distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

# MINKOWSKI
clf = KNeighborsClassifier(n_neighbors=idxs[1] + 1, metric='minkowski')
scores = cross_val_score(clf, df_x_scaled, Y, cv=folds)

fig = plt.figure()
plt.scatter(x, scores)
fig.suptitle('KNN cross validation with test size ' + str(best_testSize) + '% with minkowski distance', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))
