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
