import pandas as pd
import numpy as np
import pathlib
from sklearn import preprocessing, model_selection
from Exercise2.Datasets.Dataset import Dataset
from typing import Tuple


class Wine(Dataset):

    def __init__(self, *args, **kwargs):
        super(Wine, self).__init__(*args, **kwargs)
        self.read_csv()
        self.show_distributions("_raw")
        self.make_train_and_test_data()    
        self.remove_outliers()
        # self.show_distributions("_clean")
        self.show_correlations(figsize=(8,6))

    def read_csv(self):
        self.df = pd.read_csv(self.filepath, sep =';', encoding="ISO-8859-1")

    def make_train_and_test_data(self):
        """
           Makes train and test data and saves it in instance variables.
        """
        
        # Split into input and target variables
        target = ["quality"]
        X = self.df.drop(target, axis=1)
        Y = self.df[target]
        X_train, X_test, self.y_train, self.y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=25)

        # Scale data
        sc = preprocessing.MinMaxScaler()
        self.x_train = sc.fit_transform(X_train)
        self.x_test = sc.fit_transform(X_test)


    @staticmethod
    def get_name():
        return 'Wine'