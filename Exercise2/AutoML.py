import os
import json
from sklearn import metrics
import random

class AutoML:

    def __init__(self, dataset):
        self.dataset = dataset
        self.best_regressor = None
        self.leaderboard = {}

    def find_best_regressor(self):
        best_sol_LR = self.dataset.searchLR()  # Score
        best_sol_EN = self.dataset.full_search_EN(self.dataset.get_name())  # best_sol, all_sol, all_paths with best_sol = (params, score)
        best_sol_RF = self.dataset.full_search_RF(self.dataset.get_name())  # best_sol, all_sol, all_paths with best_sol = (params, score)
        best_sol_SVM = self.dataset.full_search_SVM(self.dataset.get_name())  # best_sol, all_sol, all_paths with best_sol = (params, score)

        self.leaderboard['LR'] = {'Score': best_sol_LR, 'Params': None}
        self.leaderboard['EN'] = {'Score': best_sol_EN[0][1], 'Params': best_sol_EN[1][0]}
        self.leaderboard['RF'] = {'Score': best_sol_RF[0][1], 'Params': best_sol_RF[1][0]}
        self.leaderboard['SVM'] = {'Score': best_sol_SVM[0][1], 'Params': best_sol_SVM[1][0]}

        maxx = min([best_sol_EN[0][1], best_sol_LR, best_sol_RF[0][1], best_sol_SVM[0][1]])
        if maxx == best_sol_LR:
            self.best_regressor = ('LR', self.dataset.calcLRPrediction())
        if maxx == best_sol_EN[0][1]:
            self.best_regressor = ('EN', self.dataset.calcENPrediction(best_sol_EN[0][0]))
        if maxx == best_sol_RF[0][1]:
            self.best_regressor = ('RF', self.dataset.calcRFPrediction(best_sol_RF[0][0]))
        if maxx == best_sol_SVM[0][1]:
            self.best_regressor = ('SVM', self.dataset.calcSVMPrediction(best_sol_SVM[0][0]))

        return self.best_regressor

    def best_prediction(self):
        reg = self.best_regressor
        if reg is None:
            reg = self.find_best_regressor()
        return reg[1]

    def best_prediction_score(self):
        return metrics.mean_squared_error(self.dataset.y_test, self.best_prediction())

    def get_leaderboard(self):
        return self.leaderboard

    def dump_leaderboard(self):
        i = random.randint()
        file = open(os.path.abspath('GD_Dump/Leaderboard_{}'.format(i)), 'w+')
        file.write(json.dumps(self.leaderboard))



