import pandas as pd
import pathlib
from sklearn import preprocessing
from Exercise2.Algorithms.linear import LinearRegression as Lra


class Purchase:

    def __init__(self):
        x_train, y_train = self.receive_train_data()
        self.xTrain = x_train
        self.yTrain = y_train
        self.xTest = self.receive_test_data()
        self.lr_prediction = None

    @staticmethod
    def receive_train_data():
        df = pd.read_csv("Datasets/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
        # Split into input and target variables
        x = df.iloc[:, 1:-1]  # Remove the ID and Class columns
        y = df.iloc[:, -1]
        # Scale data
        x_scaled = preprocessing.MinMaxScaler().fit_transform(x.values)
        return pd.DataFrame(x_scaled), y

    @staticmethod
    def receive_test_data():
        pathTest = "Datasets/purchase600-100cls-15k.tes.csv"
        dirPathTest = pathlib.Path(pathTest)
        df_test = pd.read_csv(dirPathTest)

        xTest = df_test.iloc[:, 1:]  # Remove the ID column

        x_test = xTest.values  # returns a numpy array

        x_test_scaled = preprocessing.MinMaxScaler().fit_transform(x_test)
        return pd.DataFrame(x_test_scaled)

    def get_data(self):
        return self.xTrain, self.yTrain, self.xTest

    def calcLRPrediction(self):
        lra = Lra(self.xTest, self.xTrain, self.yTrain)
        lra_prediction = lra.make_prediction()
        self.lr_prediction = lra_prediction

    def getLRPrediction(self):
        if self.lr_prediction is None:
            self.calcLRPrediction()
        return self.lr_prediction

