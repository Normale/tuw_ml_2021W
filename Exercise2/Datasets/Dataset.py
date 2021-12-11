from Exercise2.Algorithms.linear import LinearRegression as Lra
from Exercise2.Algorithms.linear import ElasticNetRegression as EN
from Exercise2.Algorithms.svm import SVM
from Exercise2.Algorithms.rf import RFR as RF
from Exercise2.GradientDescent import GradientDescent as GD
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import pandas as pd
from sklearn import model_selection , metrics
import copy
import pickle


class Dataset:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.df = None
        self.x_train = None
        self.y_train = None
        self.y_test = None
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
            plt.figure()  # prevent drawing everything on one chart
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
        dfx = pd.DataFrame(self.x_train)
        numeric = dfx.select_dtypes(include=np.number)
        non_num = dfx.select_dtypes(exclude=np.number)
        abs_z_scores = np.abs(zscore(numeric))
        filtered_entries_x = (abs_z_scores < 3).all(axis=1)
        dfx = numeric.join(non_num)

        dfy = pd.DataFrame(self.y_train)
        abs_z_scores = np.abs(zscore(dfy))
        filtered_entries_y = (abs_z_scores < 3).all(axis=1)

        filtered_entries = np.logical_and(filtered_entries_x.to_numpy(), filtered_entries_y.to_numpy())
        self.y_train = dfy[filtered_entries].to_numpy()
        self.x_train = dfx[filtered_entries].to_numpy()


        dfx = pd.DataFrame(self.x_test)
        numeric = dfx.select_dtypes(include=np.number)
        non_num = dfx.select_dtypes(exclude=np.number)
        abs_z_scores = np.abs(zscore(numeric))
        filtered_entries_x = (abs_z_scores < 3).all(axis=1)
        dfx = numeric.join(non_num)

        dfy = pd.DataFrame(self.y_test)
        abs_z_scores = np.abs(zscore(dfy))
        filtered_entries_y = (abs_z_scores < 3).all(axis=1)

        filtered_entries = np.logical_and(filtered_entries_x.to_numpy(), filtered_entries_y.to_numpy())
        self.y_test = dfy[filtered_entries].to_numpy()
        self.x_test = dfx[filtered_entries].to_numpy()

        

    def one_hot_encode(self, columns):
        # Get one hot encoding of columns
        one_hot = pd.get_dummies(self.df[columns])
        # Drop encoded columns 
        tmp = self.df.drop(columns, axis=1)
        # Join encoding 
        self.df = tmp.join(one_hot)

    # Linear Regressors
    def calcLRPrediction(self, save=True, data=None):
        if data is None:
            x_test = self.x_test
            x_train = self.x_train
            y_train = self.y_train
        else:
            x_test, x_train, y_train = data

        lra = Lra(x_test, x_train, y_train)
        lra_prediction = lra.make_prediction(x_test)
        if save is True:
            self.lr_prediction = lra_prediction
        return lra_prediction

    def calcENPrediction(self, params=None, save=True, data=None):
        if params is None:
            params = EN.params
        alpha, l1_ratio = params.values()

        if data is None:
            x_test = self.x_test
            x_train = self.x_train
            y_train = self.y_train
        else:
            x_test, x_train, y_train = data

        en = EN(x_test, x_train, y_train, alpha['value'], l1_ratio['value'])  # Params: alpha, l1_ratio
        en_prediction = en.make_prediction(x_test)
        if save is True:
            self.en_prediction = en_prediction
        return en_prediction

    # Random Forests Regressor

    def calcRFPrediction(self, params=None, save=True, data=None):
        if params is None:
            params = {'n_estimators': {'value': 50, 'type': 'int', 'min': 1, 'max': 1000, 'e': 5}}
        n = params['n_estimators']

        if data is None:
            x_test = self.x_test
            x_train = self.x_train
            y_train = self.y_train
        else:
            x_test, x_train, y_train = data

        rf = RF(x_test, x_train, y_train, n['value'])  # Params: n_estimators
        rf_prediction = rf.make_prediction(x_test)
        if save is True:
            self.rf_prediction = rf_prediction
        return rf_prediction

    # Support Vector Machine Regressor

    def calcSVMPrediction(self, params=None, save=True, data=None):
        if params is None:
            params = {'C': {'value': 1.0, 'type': 'float', 'min': 0, 'max': 10, 'e': 0.01},
                      'eps': {'value': 0.1, 'type': 'float', 'min': 0, 'max': 5, 'e': 0.001}}
        C, eps = params.values()
        kernel = 'linear'  # TODO: implement other kernels

        if data is None:
            x_test = self.x_test
            x_train = self.x_train
            y_train = self.y_train
        else:
            x_test, x_train, y_train = data

        svm = SVM(x_test, x_train, y_train, kernel=kernel, C=C, eps=eps)  # Params: C, eps, kernel
        svm_prediction = svm.make_prediction(x_test)
        if save is True:
            self.svm_prediction = svm_prediction
        return svm_prediction

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

    # Score

    def getENScore(self, x, y, p):
        X_train, X_test, Y_Train, Y_Test = model_selection.train_test_split(x, y, test_size=0.2, random_state=25)
        pred = self.calcENPrediction(params=p, save=False, data=[X_test, X_train, Y_Train])
        return metrics.mean_absolute_percentage_error(Y_Test, pred)

    # Search

    def searchEN(self, paramList=EN.params, s=0.1):
        # Initialise parameters
        x = self.x_train
        y = self.y_train
        f = self.getENScore
        gd = GD(f, paramList, x, y, s=s)
        param_sol, param_path, cost_path = gd.solve()

        best_prediction = self.calcENPrediction(param_sol)
        cost = f(x, y, paramList)
        return param_sol, cost, param_path, cost_path
        # return solution_params, best_prediction, cost

    def searchRF(self, paramList=RF.params, s=0.1):
        # Initialise parameters
        x = self.x_train
        y = self.y_train
        f = self.getRFScore
        gd = GD(f, paramList, x, y, s=s)
        param_sol, param_path, cost_path = gd.solve()

        best_prediction = self.calcENPrediction(param_sol)
        cost = f(x, y, paramList)
        return param_sol, cost, param_path, cost_path
        # return solution_params, best_prediction, cost

    def searchSVM(self, paramList=SVM.params, s=0.1):
        # Initialise parameters
        x = self.x_train
        y = self.y_train
        f = self.getSVMScore
        gd = GD(f, paramList, x, y, s=s)
        param_sol, param_path, cost_path = gd.solve()
        print(param_path)

        best_prediction = self.calcENPrediction(param_sol)
        cost = f(x, y, paramList)
        return param_sol, cost, param_path, cost_path
        # return solution_params, best_prediction, cost

    def full_search_EN(self):
        initial_search = self.searchEN()
        best_sol = (initial_search[0], initial_search[1])
        all_sol = []
        all_paths = [initial_search[2], initial_search[3]]

        gridStates_alpha = np.logspace(-2, 2, num=5, base=2).copy()
        gridStates_l1 = [0.5, 0.3]
        gridStates_l1.extend(np.logspace(-1, -3, num=3, base=10).copy())
        for s in [1, 0.1, 0.01, 0.001]:
            print("-------------------------------------")
            print("S:{}".format(s))
            print("-------------------------------------")
            for alpha in gridStates_alpha:
                for l1 in gridStates_l1:
                    par = EN.params
                    par['alpha']['value'] = alpha
                    par['l1_ratio']['value'] = l1
                    sol, cost, param_path, cost_path = self.searchEN(par, s=s)

                    all_paths.append((param_path, cost_path))

                    print(sol)
                    if cost > best_sol[1]:
                            best_sol = (copy.deepcopy(sol), cost)
                    all_sol.append((copy.deepcopy(sol), cost))
        return best_sol, all_sol, all_paths

    def full_search_RF(self):
        initial_search = self.searchRF()
        best_sol = (initial_search[0], initial_search[1])
        all_sol = []
        all_paths = [initial_search[2], initial_search[3]]

        gridStates_n = [10, 40, 160, 840]

        for s in [1, 0.1, 0.01, 0.001]:
            print("-------------------------------------")
            print("S:{}".format(s))
            print("-------------------------------------")
            for n in gridStates_n:
                par = RF.params
                par['n']['value'] = n
                sol, cost, param_path, cost_path = self.searchRF(par, s=s)

                all_paths.append((param_path, cost_path))

                print(sol)
                if cost > best_sol[1]:
                        best_sol = (copy.deepcopy(sol), cost)
                all_sol.append((copy.deepcopy(sol), cost))
        return best_sol, all_sol, all_paths

    def full_search_SVM(self):
        initial_search = self.searchSVM()
        best_sol = (initial_search[0], initial_search[1])
        all_sol = []
        all_paths = [initial_search[2], initial_search[3]]

        gridStates_c = np.logspace(-3, 3, num=7, base=2).copy()
        gridStates_eps = [0.5]
        gridStates_eps.extend(np.logspace(0, -3, num=4, base=10).copy())
        for s in [1, 0.1, 0.01, 0.001]:
            print("-------------------------------------")
            print("S:{}".format(s))
            print("-------------------------------------")
            for c in gridStates_c:
                for eps in gridStates_eps:
                    par = EN.params
                    par['c']['value'] = c
                    par['eps']['value'] = eps
                    sol, cost, param_path, cost_path = self.searchSVM(par, s=s)

                    all_paths.append((param_path, cost_path))

                    print(sol)
                    if cost > best_sol[1]:
                            best_sol = (copy.deepcopy(sol), cost)
                    all_sol.append((copy.deepcopy(sol), cost))
        return best_sol, all_sol, all_paths
