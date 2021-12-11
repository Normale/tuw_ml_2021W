from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn.ensemble import RandomForestRegressor as rfr


class RFR(Algorithm):
    params = {'n': {'value': 25, 'type': 'int', 'min': 5, 'max': 1000, 'e': 1}}

    """
    Lower C means a higher penalty on regularization
    Distance eps around prediction that is not accounted for in the score
    """
    def __init__(self, x, y, t, n=25):
        super().__init__(x, y, t)
        self.reg = rfr(n_estimators=n, random_state=123)\
            .fit(self.xTrain, self.yTrain.values.ravel())

    def get_regression(self):
        return self.reg

    def make_prediction(self, xTest):
        return self.reg.predict(xTest)

