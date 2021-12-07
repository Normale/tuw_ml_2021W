import pandas as pd
import pathlib
from sklearn import preprocessing, model_selection
from Exercise2.Datasets.Dataset import Dataset
from typing import Tuple


class Wine(Dataset):

    def __init__(self, *args, **kwargs):
        super(Wine, self).__init__(*args, **kwargs)
        self.make_train_and_test_data()        

    def make_train_and_test_data(self):
        """
           Makes train and test data and saves it in instance variables.
        """
        df = pd.read_csv(self.filepath, sep=';', encoding="ISO-8859-1")
        
        # Split into input and target variables
        target = ["quality"]
        X = df.drop(target, axis=1)
        Y = df[target]
        X_train, X_test, self.y_train, self.y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=25)

        # Scale data
        sc = preprocessing.MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        self.x_train = X_train
        self.x_test = X_test

    @staticmethod
    def get_name():
        return 'Wine'
