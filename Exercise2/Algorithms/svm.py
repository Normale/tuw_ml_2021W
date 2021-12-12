from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn import svm


class SVM(Algorithm):
    params = {'C': {'value': 100, 'type': 'float', 'min': 10**-4, 'max': 10**4, 'e': 10**-5},
              'eps': {'value': 0.1, 'type': 'float', 'min': 10**-2, 'max': 10, 'e': 10**-3}}

    """
    SCALE YOUR DATA FOR SVM!!!!
    if the data is unbalanced (e.g. many positive and few negative), set class_weight='balanced' and/or try different penalty parameters C.

    Default regression kernel is linear
    other kernels are rbf, polynomial and sigmoid

    Lower C means a higher penalty on regularization
    Distance eps around prediction that is not accounted for in the score
    """
    def __init__(self, x, y, t, kernel='linear', C=1.0, eps=0.1):
        super().__init__(x, y, t)
        self.kernel = kernel
        self.reg = svm.SVR(kernel=kernel, C=C, epsilon=eps).fit(self.xTrain, self.yTrain.ravel())

    def make_regression(self, kernel='linear', C=1.0, eps=0.1):
        self.kernel = kernel
        self.reg = svm.SVR(kernel=kernel, C=C, epsilon=eps).fit(self.xTrain, self.yTrain.ravel())

    def get_regression(self):
        return self.reg

    def make_prediction(self, xTest):
        return self.reg.predict(xTest)  # LR(positive=True) for only positive coef
