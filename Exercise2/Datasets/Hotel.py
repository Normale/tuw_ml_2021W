import pandas as pd
import numpy as np
import pathlib
from sklearn import preprocessing, model_selection
from Exercise2.Datasets.Dataset import Dataset
from typing import Tuple


class Hotel(Dataset):

    def __init__(self, *args, **kwargs):
        self.columns_mapping = {
            "Price in Millions": 'Price'
        }
        super(Hotel, self).__init__(*args, **kwargs)
        self.read_csv()
        self.show_distributions("_raw")
        self.remove_outliers()
        self.show_distributions("_clean")
        self.one_hot_encode(["City"])
        self.make_train_and_test_data()
        self.show_correlations(figsize=(8,6))

    def read_csv(self):
        self.df = pd.read_csv(self.filepath, encoding="ISO-8859-1")
        self.df.rename(columns=self.columns_mapping, inplace=True)

    
    def make_train_and_test_data(self):
        """
           Makes train and test data and saves it in instance variables.
        """
        # Split into input and target variables
        target = ["Price"]
        X = self.df.drop(target, axis=1)
        Y = self.df[target]
        X_train, X_test, self.y_train, self.y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=25)

        # Scale data
        sc = preprocessing.MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        self.x_train = X_train
        self.x_test = X_test



    @staticmethod
    def get_name():
        return 'Housing'