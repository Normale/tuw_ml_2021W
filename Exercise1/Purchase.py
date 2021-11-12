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


# Use the k-NN method (up to k_max) to predict the output variable on x_test, using the training data
def predict_knn(x_test, x_train, y_train, k_min=1, k_max=25):
    all_predictions = []
    for k in tqdm(range(k_min,k_max)):
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k,p=2)
        # Fit the classifier to the data
        knn.fit(x_train, y_train)
        # Predict on x_test
        prediction = knn.predict(x_test)
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


df = pd.read_csv("Datasets/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
print(df.head())


X = df.iloc[:, 1:-1]  # Remove the ID and Class columns
Y = df.iloc[:, -1]

# x = X.values # returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# #scaler = preprocessing.StandardScaler()
# #x_scaled = scaler.fit_transform(x)
df_x_scaled = pd.DataFrame(X)
print(df_x_scaled)

#################################################################
# ON TEST DATA
test_data = pd.read_csv("Datasets/purchase600-100cls-15k.tes.csv", encoding="ISO-8859-1")
df_test = pd.DataFrame(test_data)
test_x = df_test.iloc[:, 1:]  # Remove the ID column
df_test_normalized = test_x

print("df_test_normalized: " , df_test_normalized)

# Training the different algorithms
X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, Y, test_size=0.20, random_state=35)

# # KNN
# all_predictions = predict_knn(X_test, X_train, Y_train, 1, 30)
# results = check_accuracy(Y_test, all_predictions)
# print("\nTRAINING USING KNN")
# max_value = max(results)
# max_index = results.index(max_value)
# print("Max value:", max_value)
# print("Max index:", max_index)

# # RANDOM FORREST
# all_predictions = predict_random_forrest(X_test, X_train, Y_train, 1, 100)
# results = check_accuracy(Y_test, all_predictions)
# print("\nTRAINING USING RANDOM FORREST")
# max_value = max(results)
# max_index = results.index(max_value)
# print("Max value:", max_value)
# print("Max index:", max_index)
#
# # LOGISTIC REGRESSION
# all_predictions = predict_logistic_regression(X_test, X_train, Y_train)
# results = check_accuracy(Y_test, all_predictions)
# print("\nTRAINING USING KNN")
# print(results)
#
# # LINEDAR DISCRIMINANT ANALYSIS
# #all_predictions = predict_lda(X_test, X_train, Y_train)
# #results = check_accuracy(Y_test, all_predictions)
# #print("\nTRAINING USING LDA")
# #print(results)
#
# # QUADRATIC DISCRIMINANT ANALYSIS
# all_predictions = predict_qda(X_test, X_train, Y_train)
# results = check_accuracy(Y_test, all_predictions)
# print("\nTRAINING USING QDA")
# print(results)

# SUPPORT VECTOR MACHINES
all_predictions = predict_svm(X_test, X_train, Y_train)
results = check_accuracy(Y_test, all_predictions)
print("\nTRAINING USING SVM")
print(results)

# # NAIVE BAYES
# all_predictions = predict_naive_bayes(X_test, X_train, Y_train)
# results = check_accuracy(Y_test, all_predictions)
# print("\nTRAINING USING NAIVE BAYES")
# print(results)

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
svm = svm.SVC(kernel='poly', random_state=42)
svm.fit(df_x_scaled, Y)
prediction = svm.predict(df_test_normalized)
data = {'ID': df_test.iloc[:, 0], 'class': prediction}
output = pd.DataFrame(data, columns=['class'], index=data['ID'])

# Save
path = 'lr_purchase.csv'
dirPath = pathlib.Path(path)
output= output.to_csv(dirPath)
print(output)