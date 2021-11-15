import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from time import time

# Use the k-NN method (up to k_max) to predict the output variable on x_test, using the training data
def predict_knn(x_test, x_train, y_train, k_min=1, k_max=25):
    all_predictions = []
    # Create KNN classifier
    for k in tqdm(range(k_min, k_max)):
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k,p=1)
        # Fit the classifier to the data
        t1 = time()
        knn.fit(x_train, y_train)
        # Predict on x_test
        t2 = time()
        prediction = knn.predict(x_test)
        t3 = time()
        all_predictions.append(prediction)

    return all_predictions

def predict_logistic_regression(x_test, x_train, y_train):
    all_predictions = []
    lr = LogisticRegression(random_state = 0)
    lr.fit(x_train, y_train)
    prediction = lr.predict(x_test)
    all_predictions.append(prediction)
    return all_predictions

def predict_lda(x_test, x_train, y_train):
    all_predictions = []
    lda = LDA()
    lda.fit(x_train,y_train)
    prediction = lda.predict(x_test)
    all_predictions.append(prediction)
    return all_predictions

def predict_random_forrest(x_test, x_train, y_train, k_min=10, k_max=100):
    all_predictions = []
    grid_params = {
        'criterion': ["gini", "entropy"],
        'max_depth': range(10,100,5),
        'n_estimators': range(10,200,10)
    }
    gs = GridSearchCV(
        RandomForestClassifier(),
        grid_params,
        verbose = 3,
        cv = 5,
        n_jobs = -1
    )
    gs_results = gs.fit(X_train, Y_train)
    print(f"RESULTS: \n {gs_results}")
    print(f"{gs_results.best_score_=}, {gs_results.best_estimator_=}, {gs_results.best_params_=}")
    return
    for k in tqdm(range(k_min,k_max)):
        rndf = RandomForestClassifier(n_estimators=k)
        rndf.fit(x_train,y_train)
        prediction = rndf.predict(x_test)
        all_predictions.append(prediction)
    return all_predictions

def predict_qda(x_test, x_train, y_train):
    all_predictions = []
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    prediction = qda.predict(x_test)
    all_predictions.append(prediction)
    return all_predictions

def predict_svm(x_test, x_train, y_train):
    all_predictions = []
    svc = svm.SVC()
    svc.fit(x_train, y_train)
    prediction = svc.predict(x_test)
    all_predictions.append(prediction)
    return all_predictions

def predict_naive_bayes(x_test, x_train, y_train):
    all_predictions = []
    gnb = GaussianNB()
    prediction = gnb.fit(X_train, Y_train).predict(X_test)
    all_predictions.append(prediction)
    return all_predictions

# Check the accuracy of given predictions on the test set y_test
def check_accuracy(y_test, predictions):
    ground_truth = y_test.to_list()
    size = len(ground_truth)
    results = []

    for predict in predictions:
        count = 0
        for i, x in enumerate(ground_truth):
            if predict[i] == ground_truth[i]:
                count += 1
        results.append(count / size)
    return results


#############################################################



df = pd.read_csv("Datasets/speeddating_1.csv", encoding="ISO-8859-1")
print(df.head())
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
print(encoded['met'].value_counts())
encoded.iloc[5550:5560] 
print(encoded.iloc[2420:2430])
cat_1hot = cat_1hot.join(encoded)
df[df.isna().any(axis=1)]
cat_1hot = cat_1hot[~cat_1hot.isna().any(axis=1)] #remove all (8) rows containing nulls

from sklearn.model_selection import train_test_split
train, test = train_test_split(cat_1hot, test_size=0.1)
Y_train = train.iloc[:, -1]
X_train = train.iloc[:, 0:-1]  # Remove the ID and Class columns
X_test = test.iloc[:, 0:-1]  # Remove the ID and Class columns
Y_test = test.iloc[:, -1]

# KNN
all_predictions = predict_knn(X_test, X_train, Y_train, 1, 30)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING KNN")
max_value = max(results)
max_index = results.index(max_value)
print("Max value:", max_value)
print("Max index:", max_index)
print("Time knn:", time_knn)


# RANDOM FORREST
all_predictions = predict_random_forrest(X_test, X_train, Y_train, 1, 100)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING RANDOM FORREST")
max_value = max(results)
max_index = results.index(max_value)
print("Max value:", max_value)
print("Max index:", max_index)

raise "END"

# LOGISTIC REGRESSION
all_predictions = predict_logistic_regression(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING KNN")
print(results)

# LINEDAR DISCRIMINANT ANALYSIS
all_predictions = predict_lda(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING LDA")
print(results)

# QUADRATIC DISCRIMINANT ANALYSIS
all_predictions = predict_qda(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING QDA")
print(results)

# SUPPORT VECTOR MACHINES
all_predictions = predict_svm(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING SVM")
print(results)

# NAIVE BAYES
all_predictions = predict_naive_bayes(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING NAIVE BAYES")
print(results)

# # TEST FOR KAGGLE (KNN)
# knn = KNeighborsClassifier(n_neighbors=27,p=1)
# knn.fit(df_x_scaled, Y)
# prediction = knn.predict(df_test_normalized)
#
# data = {'ID': df_test.iloc[:, 0], 'class': prediction}
# output = pd.DataFrame(data, columns=['class'], index=data['ID'])
# print(output)

# # TEST FOR KAGGLE (LR)
# LR = LogisticRegression(random_state = 0)
# LR.fit(df_x_scaled, Y)
# prediction = LR.predict(df_test_normalized)
#
# data = {'ID': df_test.iloc[:, 0], 'class': prediction}
# output = pd.DataFrame(data, columns=['class'], index=data['ID'])
# print(output)

# TEST FOR KAGGLE (SVM)
# svm = svm.SVC(kernel='poly', random_state=42)
# svm.fit(df_x_scaled, Y)
# prediction = svm.predict(df_test_normalized)
# data = {'ID': df_test.iloc[:, 0], 'class': prediction}
# output = pd.DataFrame(data, columns=['class'], index=data['ID'])

# Save
# path = 'lr_purchase.csv'
# dirPath = pathlib.Path(path)
# output= output.to_csv(dirPath)
# print(output)