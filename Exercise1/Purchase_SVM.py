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


def predict_svm(x_test, x_train, y_train):
    all_predictions = []
    svc = svm.SVC()
    svc.fit(x_train, y_train)
    prediction = svc.predict(x_test)
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
df_x = pd.DataFrame(X)
print(df_x)

#################################################################
# ON TEST DATA
test_data = pd.read_csv("Datasets/purchase600-100cls-15k.tes.csv", encoding="ISO-8859-1")
df_test = pd.DataFrame(test_data)
test_x = df_test.iloc[:, 1:]  # Remove the ID column

print("test_x: ", test_x)

# Training the different algorithms
for testSize in range(5,16,1):
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, Y, test_size=testSize/100, random_state=35)

    # SUPPORT VECTOR MACHINES
    all_predictions = predict_svm(X_test, X_train, Y_train)
    results = check_accuracy(Y_test, all_predictions)
    print("\nTRAINING USING SVM")
    print("\nTest size = ", testSize/100)
    print(results)

    # TEST FOR KAGGLE (SVM)
    SVM = svm.SVC(kernel='linear', random_state=42)
    SVM.fit(df_x, Y)
    prediction = SVM.predict(test_x)
    data = {'ID': df_test.iloc[:, 0], 'class': prediction}
    output = pd.DataFrame(data, columns=['class'], index=data['ID'])

# # Save
# path = 'lr_purchase.csv'
# dirPath = pathlib.Path(path)
# output= output.to_csv(dirPath)
# print(output)