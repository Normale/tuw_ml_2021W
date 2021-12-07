from Exercise2.Algorithms.Algorithm import Algorithm
from sklearn import svm


class SVM(Algorithm):

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
        print("---   INITIALIZING (SVM) with " + kernel + " kernel ---")
        self.reg = svm.SVR(kernel=kernel, C=C, epsilon=eps)\
            .fit(self.xTrain, self.yTrain.values.ravel())  # Ravel makes it a 1xN vector

    def make_regression(self, kernel='linear', C=1.0, eps=0.1):
        self.kernel = kernel
        self.reg = svm.SVR(kernel=kernel, C=C, epsilon=eps)\
            .fit(self.xTrain, self.yTrain.values.ravel())  # Ravel makes it a 1xN vector

    def get_regression(self):
        return self.reg

    def make_prediction(self):
        print("---   PREDICTING (SVM) with kernel " + self.kernel + "  ---")
        return self.reg.predict(self.xTest)  # LR(positive=True) for only positive coef

    def get_R2(self, yTest):
        return self.reg.score(self.xTest, yTest)
