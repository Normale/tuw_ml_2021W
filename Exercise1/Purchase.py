import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px


# Use the k-NN method (up to k_max) to predict the output variable on x_test, using the training data
def predict(x_test, x_train, y_train, k_min=1, k_max=25):
    all_predictions = []
    for k in tqdm(range(k_min,k_max)):
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit the classifier to the data
        knn.fit(x_train, y_train)
        # Predict on x_test
        prediction = knn.predict(x_test)
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


# Split our training data into training and test data to find the best k

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=35)
#
# all_predictions = predict(X_test, X_train, Y_train, 1, 25)
# results = check_accuracy(Y_test, all_predictions)
#
# print(results)

#################################################################
# ON TEST DATA
test_data = pd.read_csv("Datasets/purchase600-100cls-15k.tes.csv", encoding="ISO-8859-1")
test_x = test_data.iloc[:, 1:]  # Remove the ID column

all_predictions = predict(test_x, X, Y, 16, 17)
# print(all_predictions[0])
data = {'ID': test_data.iloc[:, 0], 'class': all_predictions[0]}
output = pd.DataFrame(data, columns=['class'], index=data['ID'])
output.to_csv('./output.csv')
