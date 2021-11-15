# -*- coding: utf-8 -*-
"""Purchase_SVM.ipynb
"""

import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
import timeit
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt


# ---------------- DEFINE FUNCTIONS ----------------

# Train the SVM on the training data and predict using the test data
def predict_svm(x_te, x_tr, y_tr):
    every_predictions = []
    grid_params = { 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(3,5),
        'decision_function_shape': ['ovo', 'ovr'],
        'gamma': ['scale', 'auto'],
        'C': [0.5, 1, 1.5, 5],
        'shrinking': [True, False],
        'coef0': [0, 0.5, -1]
    }
    gs = GridSearchCV(
        svm.SVC(),
        grid_params,
        verbose = 3,
        cv = 3,
        n_jobs = -1
    )    
    gs_results = gs.fit(X_train, Y_train)
    print(f"RESULTS: \n {gs_results}")
    print(f"{gs_results.best_score_=}, {gs_results.best_estimator_=}, {gs_results.best_params_=}")
    return 
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
df = pd.read_csv("Datasets/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
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

print("\nTRAINING USING SVM")
for testSize in range(5, 17, 2):
    X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=testSize/100, random_state=35)

    # SUPPORT VECTOR MACHINES
    start = timeit.default_timer()
    all_predictions = predict_svm(X_test, X_train, Y_train)
    stop = timeit.default_timer()
    time = stop-start
    print('Time: ', time)
    results = check_accuracy(Y_test, all_predictions)
    print("\nTest size = ", testSize/100)
    print(results)
    testSize = testSize/100
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

cv_ = int((100/(best_testSize*100)))
testSize = 100/cv_

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, Y, cv=cv_)
x = range(1,cv_+1)

fig = plt.figure()
plt.scatter(x,scores)
fig.suptitle('SVM cross validation with test size ' + str(testSize) + '% with linear kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

cv_ = int((100/(best_testSize*100)))

clf = svm.SVC(kernel='rbf', C=1, random_state=42)
scores = cross_val_score(clf, X, Y, cv=cv_)
x = range(1,cv_+1)

fig = plt.figure()
plt.scatter(x,scores)
fig.suptitle('SVM cross validation with test size ' + str(testSize) + '% with rbf kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))

cv_ = int((100/(best_testSize*100)))

clf = svm.SVC(kernel='poly', C=1, random_state=42)
scores = cross_val_score(clf, X, Y, cv=cv_)
x = range(1,cv_+1)

fig = plt.figure()
plt.scatter(x,scores)
fig.suptitle('SVM cross validation with test size ' + str(testSize) + '% with polynomial kernel', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
print("Max accuracy = ", np.max(scores))
