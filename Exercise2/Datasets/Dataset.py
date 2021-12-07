from Exercise2.Algorithms.linear import LinearRegression as Lra
from Exercise2.Algorithms.linear import ElasticNetRegression as EN
from Exercise2.Algorithms.svm import SVM
from Exercise2.Algorithms.rf import RFR as RF
from pathlib import Path


class Dataset:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.lr_prediction = None
        self.en_prediction = None
        self.svm_prediction = None
        self.rf_prediction = None

    def get_data(self):
        return self.x_train, self.y_train, self.x_test

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