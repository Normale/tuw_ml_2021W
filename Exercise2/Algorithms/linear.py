from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn.linear_model import LinearRegression as LR


class LinearRegression(Algorithm):

    def __init__(self, x, y, t):
        Algorithm.__init__(self, x, y, t)
        self.reg = LR().fit(self.xTrain, self.yTrain)

    def get_regression(self):
        return self.reg

    def make_prediction(self):
        return LR().fit(self.xTrain, self.yTrain).predict(self.xTest) # LR(positive=True) for only positive coef
