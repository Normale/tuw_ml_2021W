from Exercise2.Algorithms.linear import LinearRegression as Lra
from Exercise2.Algorithms.linear import ElasticNetRegression as EN
from Exercise2.Algorithms.svm import SVM
from Exercise2.Algorithms.rf import RFR as RF
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import pandas as pd

class Dataset:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.df = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.lr_prediction = None
        self.en_prediction = None
        self.svm_prediction = None
        self.rf_prediction = None
        Path(f"Graphs/{self.__class__.__name__}/distributions").mkdir(parents=True, exist_ok=True)

    def get_data(self):
        return self.x_train, self.y_train, self.x_test

    def show_distributions(self,n=None):
        plt.ioff()
        for col in self.df.columns:
            plt.figure() #prevent drawing everything on one chart
            temp = self.df[col]
            if is_numeric_dtype(temp):
                plot = temp.hist(bins=30, log=True)
            else:
                plot = temp.value_counts().plot.pie(autopct='%.1f%%')
            fig = plot.get_figure()
            fig.savefig(f"Graphs/{self.__class__.__name__}/distributions/{col}{n}.png")
            fig.data = []
            plt.close(fig)
        plt.ion()

    def show_correlations(self, figsize=(6,5)):
        plt.ioff()
        plt.figure(figsize=figsize)
        corr = self.df.corr().round(2)
        plot = sns.heatmap(corr, annot=True,
            cmap=sns.diverging_palette(20, 220, n=200))
        fig = plot.get_figure()
        plt.tight_layout()
        fig.savefig(f"Graphs/{self.__class__.__name__}/correlations.png")
        plt.close(fig)
        plt.ion()

    def remove_outliers(self):
        numeric = self.df.select_dtypes(include=np.number)
        non_num = self.df.select_dtypes(exclude=np.number)
        z_scores = zscore(numeric)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        numeric = numeric[filtered_entries]
        numeric = numeric.join(non_num)
        self.df = numeric

    def one_hot_encode(self, columns):
        # Get one hot encoding of columns
        one_hot = pd.get_dummies(self.df[columns])
        # Drop encoded columns 
        tmp = self.df.drop(columns,axis = 1)
        # Join encoding 
        self.df = tmp.join(one_hot)

    # Linear Regressors
    def calcLRPrediction(self):
        lra = Lra(self.x_test, self.x_train, self.y_train)
        lra_prediction = lra.make_prediction()
        self.lr_prediction = lra_prediction

    def calcENPrediction(self, alpha=0.5, l1_ratio=0.5):
        en = EN(self.x_test, self.x_train, self.y_train, alpha, l1_ratio)  # Params: alpha, l1_ratio
        en_prediction = en.make_prediction()
        self.en_prediction = en_prediction

    # Random Forests Regressor

    def calcRFPrediction(self, n=50):
        rf = RF(self.x_test, self.x_train, self.y_train, n)  # Params: n_estimators
        rf_prediction = rf.make_prediction()
        self.rf_prediction = rf_prediction

    # Support Vector Machine Regressor

    def calcSVMPrediction(self, kernel='linear', C=1.0, eps=0.1):
        svm = SVM(self.x_test, self.x_train, self.y_train, kernel=kernel, C=C, eps=eps)  # Params: C, eps, kernel
        svm_prediction = svm.make_prediction()
        self.svm_prediction = svm_prediction

    # Getters

    def getLRPrediction(self):
        if self.lr_prediction is None:
            self.calcLRPrediction()
        return self.lr_prediction

    def getENPrediction(self):
        if self.en_prediction is None:
            self.calcENPrediction()
        return self.en_prediction

    def getRFPrediction(self):
        if self.rf_prediction is None:
            self.calcRFPrediction()
        return self.rf_prediction

    def getSVMPrediction(self):
        if self.svm_prediction is None:
            self.calcSVMPrediction()
        return self.svm_prediction

    # Setters

    def setLRPrediction(self, pred):
        self.lr_prediction = pred

    def setENPrediction(self, pred):
        self.en_prediction = pred

    def setRFPrediction(self, pred):
        self.rf_prediction = pred

    def setSVMPrediction(self, pred):
        self.svm_prediction = pred