
class AutoML:

    def __init__(self, dataset):
        self.dataset = dataset
        self.best_regressor = None

    def find_best_regressor(self):
        best_sol_LR = self.dataset.search_LR()
        best_sol_EN = self.dataset.full_search_EN()
        best_sol_RF = self.dataset.full_search_RF()
        best_sol_SVM = self.dataset.full_search_SVM()
