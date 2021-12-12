from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn import linear_model as LM


class LinearRegression(Algorithm):  # Upgrade to LinearRegressionCV for automatic CV!

    def __init__(self, x, y, t):
        super().__init__(x, y, t)
        self.reg = LM.LinearRegression().fit(self.xTrain, self.yTrain)

    def get_regression(self):
        return self.reg

    def make_prediction(self, xTest):
        return self.reg.predict(xTest)  # LR(positive=True) for only positive coef


class ElasticNetRegression(Algorithm):  # Upgrade to ElasticNetCV for automatic CV!

    params = {'alpha': {'value': 0.5, 'type': 'float', 'min': 10**-8, 'max': 20, 'e': 10**-10},
              'l1_ratio': {'value': 0.05, 'type': 'float', 'min': 10**-6, 'max': 1, 'e': 10**-10}}
    # TODO: Iterate alpha logarithmically

    """
    Elastic Net Regression combines the best of both worlds from Ridge Regression and Lasso Regression
    The parameter alpha multiplies the penalty terms (=0 mean OLS)
    The parameter l1_ratio is the mixing parameter between 0 and 1:
    =0 means L2 (ridge) and =1 means L1 (Lasso)
    """
    def __init__(self, x, y, t, alpha=0.5, l1_ratio=0.5):
        Algorithm.__init__(self, x, y, t)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.reg = LM.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=123).fit(self.xTrain, self.yTrain)

    def get_regression(self):
        return self.reg

    def make_prediction(self, xTest):
        return self.reg.predict(xTest)
