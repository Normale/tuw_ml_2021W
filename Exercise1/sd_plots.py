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

def predict_knn(x_test, x_train, y_train, k=1):
    knn = KNeighborsClassifier(n_neighbors=k,p=1)
    # Fit the classifier to the data
    t1 = time()
    knn.fit(x_train, y_train)
    # Predict on x_test
    t2 = time()
    prediction = knn.predict(x_test)
    t3 = time()        
    return prediction, t2-t1, t3-t2

def predict_random_forrest(x_test, x_train, y_train,k=20):
    rndf = RandomForestClassifier(n_estimators=k)
    t1 = time()
    rndf.fit(x_train,y_train)
    t2 = time()
    prediction = rndf.predict(x_test)
    t3 = time()
    return prediction, t2-t1, t3-t2

def predict_svm(x_test, x_train, y_train):
    all_predictions = []
    svc = svm.SVC()
    t1 = time()
    svc.fit(x_train, y_train)
    t2 = time()
    prediction = svc.predict(x_test)
    t3 = time()
    return prediction, t2-t1, t3-t2

# Check the accuracy of given predictions on the test set y_test
def check_accuracy(y_test, predictions):
    ground_truth = y_test.to_list()
    size = len(ground_truth)
    count = 0
    for i, x in enumerate(ground_truth):
        if predictions[i] == ground_truth[i]:
            count += 1
    result = (count / size)
    return result

#Preprocessing

# df = pd.read_csv("Datasets/speeddating_1.csv", encoding="ISO-8859-1")
# print(df.head())
# filter_col = [col for col in df if col.startswith('d_')]

# cols = set(df.columns)
# categorized_cols = ["gender"] + [col for col in df if col.startswith('d_')]
# categorized_cols.remove('d_age')
# categorized_df = df[categorized_cols]

# cat_1hot = pd.get_dummies(categorized_df)
# encoded = df[["samerace", "met", "match"]]
# encoded = encoded.replace('?', 0)
# encoded = encoded.apply(pd.to_numeric)
# a = (0, 1)
# encoded = encoded[encoded['met'].isin(a)]
# print(encoded['met'].value_counts())
# encoded.iloc[5550:5560] 
# print(encoded.iloc[2420:2430])
# cat_1hot = cat_1hot.join(encoded)
# df[df.isna().any(axis=1)]
# cat_1hot = cat_1hot[~cat_1hot.isna().any(axis=1)] #remove all (8) rows containing nulls

# from sklearn.model_selection import train_test_split
# train, test = train_test_split(cat_1hot, test_size=0.1)
# Y_train = train.iloc[:, -1]
# X_train = train.iloc[:, 0:-1]  # Remove the ID and Class columns
# X_test = test.iloc[:, 0:-1]  # Remove the ID and Class columns
# Y_test = test.iloc[:, -1]


# # KNN
# print("Training KNN..")
# all_predictions, time_knn_t, time_knn_p = predict_knn(X_test, X_train, Y_train, 5)
# result_knn = check_accuracy(Y_test, all_predictions)

# # RANDOM FORREST
# print("Training RF..")
# all_predictions, time_rf_t, time_rf_p = predict_random_forrest(X_test, X_train, Y_train, 20)
# result_rf = check_accuracy(Y_test, all_predictions)

# # SUPPORT VECTOR MACHINES
# print("Training SVM..")
# all_predictions, time_svm_t, time_svm_p = predict_svm(X_test, X_train, Y_train)
# result_svm = check_accuracy(Y_test, all_predictions)



# print(f"""
# time knn training: {time_knn_t}s \tpredicting{time_knn_p}s\t accuracy: {result_knn}\n
# time rf training: {time_rf_t}s  \tpredicting{time_knn_p}s accuracy: {result_rf}\n
# time svm training: {time_svm_t}s \tpredicting{time_knn_p}s accuracy: {result_svm}\n
# """)


# # Read the data
# df2 = pd.read_csv("Datasets/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
# # print(df.head())

# # Split into input and target variables
# X2 = df2.iloc[:, 1:-1]  # Remove the ID and Class columns
# Y2 = df2.iloc[:, -1]

# # Scale data
# x2 = X2.values  # returns a numpy array
# min_max_scaler2 = preprocessing.MinMaxScaler()
# x_scaled2 = min_max_scaler2.fit_transform(x2)
# df_x_scaled2 = pd.DataFrame(x_scaled2)
# # print(df_x_scaled)

# # Import test-data and scaling the data
# pathTest2 = "Datasets/purchase600-100cls-15k.tes.csv"
# dirPathTest2 = pathlib.Path(pathTest2)
# df_test2 = pd.read_csv(dirPathTest2)

# xTest2 = df2.iloc[:, 1:-1]  # Remove the ID and Class columns
# x_test2 = xTest2.values  # returns a numpy array

# x_test_scaled2 = min_max_scaler2.fit_transform(x_test2)
# df_test_normalized2 = pd.DataFrame(x_test_scaled2)
# # print(df_test_normalized)
# testSize2 = 0.3
# X2_train, X2_test, Y2_train, Y2_test = train_test_split(df_x_scaled2, Y2, test_size=testSize2 / 100, random_state=35)


# # KNN
# print("Training KNN..")
# all_predictions, time2_knn_t, time2_knn_p = predict_knn(X2_test, X2_train, Y2_train, 5)
# result2_knn = check_accuracy(Y2_test, all_predictions)

# # RANDOM FORREST
# print("Training RF..")
# all_predictions, time2_rf_t, time2_rf_p = predict_random_forrest(X2_test, X2_train, Y2_train, 20)
# result2_rf = check_accuracy(Y2_test, all_predictions)

# # SUPPORT VECTOR MACHINES
# print("Training SVM..")
# all_predictions, time2_svm_t, time2_svm_p = predict_svm(X2_test, X2_train, Y2_train)
# result2_svm = check_accuracy(Y2_test, all_predictions)


# print(f"""
# time knn training: {time2_knn_t}s \tpredicting{time2_knn_p}s\t accuracy: {result2_knn}\n
# time rf training: {time2_rf_t}s  \tpredicting{time2_knn_p}s accuracy: {result2_rf}\n
# time svm training: {time2_svm_t}s \tpredicting{time2_knn_p}s accuracy: {result2_svm}\n
# """)





x = np.arange(3)
labels = ["KNN_1", "RANDOM FOREST_1", "SVM_1"]
# times_t = [time_knn_t, time_rf_t, time_svm_t]
times_t = [0.006, 0.25, 3.5]
# times2_t = [time2_knn_t, time2_rf_t, time2_svm_t]
times2_t = [0.014, 1.6, 32]
width = 0.40

plt.xticks(x, labels = labels)
plt.xlabel("Algorithms")
plt.ylabel("Time")
plt.legend(["dating", "purchases"])
plt.bar(x-width/2, times_t, width)
plt.bar(x+width/2, times2_t, width, color = 'green')
plt.show()