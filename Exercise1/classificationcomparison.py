# -*- coding: utf-8 -*-
"""ClassificationComparison.ipynb
"""

from sklearn import model_selection
from sklearn.svm import SVC
import pandas as pd
import pathlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import plot_confusion_matrix
import numpy as np

# Load the data
path = 'Datasets/breast-cancer-diagnostic.shuf.lrn.csv'
dirPath = pathlib.Path(path)
df = pd.read_csv(dirPath)
# print(df.head())

# Set class-label from true/false to 0/1
df['class'] = df['class'].astype(int)

# Split into input and target variables
X = df.iloc[:, 2:]  # Remove the ID and Class columns
Y = df.iloc[:, 1]

# Scale data
x = X.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
# scaler = preprocessing.StandardScaler()
# x_scaled = scaler.fit_transform(x)
x_scaled = min_max_scaler.fit_transform(x)
df_x_scaled = pd.DataFrame(x_scaled)
# print(df_x_scaled)

# prepare models
models = [('KNN', KNeighborsClassifier(n_neighbors=4, metric='euclidean')),
          ('RANDOM FORREST', RandomForestClassifier(n_estimators=10)), ('SVM', SVC(kernel='linear', C=1))]
# evaluate each model in turn
results = []
names = []
seed = 7
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=4, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.show()

# CHECK FOR IMBALANCE IN THE DATASET
len_true = len(Y[Y == 1])
len_false = len(Y[Y == 0])
print("IS THE DATASET IMBALANCED?\n")
print("Number of true: ", len_true)
print("Number of false: ", len_false)

trainX, testX, trainy, testy = train_test_split(df_x_scaled, Y, test_size=0.25, random_state=2)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
model = svm.SVC(kernel='linear', C=1, random_state=42, probability=True)
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
print(lr_probs)
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# SVM TEST
X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=0.20, random_state=35)

svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train)
prediction = svc.predict(X_test)
plot_confusion_matrix(svc, X_test, Y_test)
plt.show()
