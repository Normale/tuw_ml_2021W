from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn import linear_model as LM


class LinearRegression(Algorithm):  # Upgrade to LinearRegressionCV for automatic CV!

    def __init__(self, x, y, t):
        super().__init__(x, y, t)
        print("---   INITIALIZING (LR)   ---")
        print(self.yTrain)
        print(self.yTrain.values.ravel())
        self.reg = LM.LinearRegression().fit(self.xTrain, self.yTrain.values.ravel())  # Ravel makes it a 1xN vector

    def get_regression(self):
        return self.reg

    def make_prediction(self):
        print("---   PREDICTING (LR)   ---")
        return self.reg.predict(self.xTest)  # LR(positive=True) for only positive coef

    def get_R2(self, yTest):
        return self.reg.score(self.xTest, yTest)


class ElasticNetRegression(Algorithm):  # Upgrade to ElasticNetCV for automatic CV!

    alpha_range = (0, 20)  # Iterate logarithmically!
    l1_ratio_range = (0, 1)

    """
    Elastic Net Regression combines the best of both worlds from Ridge Regression and Lasso Regression
    The parameter alpha multiplies the penalty terms (=0 mean OLS)
    The parameter l1_ratio is the mixing parameter between 0 and 1:
    =0 means L2 (ridge) and =1 means L1 (Lasso)
    """
    def __init__(self, x, y, t, alpha=0.5, l1_ratio=0.5):
        Algorithm.__init__(self, x, y, t)
        print("---   INITIALIZING (EN)   ---")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.reg = LM.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=123)\
            .fit(self.xTrain, self.yTrain.values.ravel())  # Ravel makes it a 1xN vector

    def get_regression(self):
        return self.reg

    def make_prediction(self):
        print("---   PREDICTING (EN)   ---")
        return self.reg.predict(self.xTest)

    def get_R2(self, yTest):
        return self.reg.score(self.xTest, yTest)
