import pandas as pd
import pathlib
from sklearn import preprocessing
from Exercise2.Datasets.Dataset import Dataset


class Purchase(Dataset):

    def __init__(self):
        super().__init__()
        x_train, y_train = self.make_train_data()
        self.xTrain = x_train
        self.yTrain = y_train
        self.xTest = self.make_test_data()

    """
    Here the dataset fetches and preprocesses its Training Data!
    """
    @staticmethod
    def make_train_data():
        df = pd.read_csv("Datasets/Raw/purchase600-100cls-15k.lrn.csv", encoding="ISO-8859-1")
        # Split into input and target variables
        x = df.iloc[:, 1:-1]  # Remove the ID and Class columns
        y = df.iloc[:, -1]
        # Scale data
        x_scaled = preprocessing.MinMaxScaler().fit_transform(x.values)
        return pd.DataFrame(x_scaled), y

    """
    Here the dataset fetches and preprocesses its Test Data!
    """
    @staticmethod
    def make_test_data():
        pathTest = "Datasets/Raw/purchase600-100cls-15k.tes.csv"
        dirPathTest = pathlib.Path(pathTest)
        df_test = pd.read_csv(dirPathTest)

        xTest = df_test.iloc[:, 1:]  # Remove the ID column

        x_test = xTest.values  # returns a numpy array

        x_test_scaled = preprocessing.MinMaxScaler().fit_transform(x_test)
        return pd.DataFrame(x_test_scaled)

    @staticmethod
    def get_name():
        return 'Purchase'
