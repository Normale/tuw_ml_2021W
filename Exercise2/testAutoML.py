from sklearn.model_selection import RepeatedKFold
from Exercise2.Datasets.Wine import Wine
from Exercise2.Datasets.Concrete import Concrete
from Exercise2.Datasets.Hotel import Hotel
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
import numpy as np
from tpot import TPOTRegressor
import tpot
import autosklearn.regression
import sklearn.metrics

# Temporary solution due to discrepancies between IDEs
if "Exercise2" not in os.getcwd():
    if "Exercise2" in os.listdir(os.getcwd()):
        os.chdir("Exercise2")
    else:
        raise NotImplementedError
# ----- TPOT SOLUTION -----
datasets = []
filepath = Path("Datasets/Raw") / "winequality-red.csv"
datasets.append(Wine(filepath))
fp2 = Path("Datasets/Raw") / "ConcreteStrengthData.csv"
datasets.append(Concrete(fp2))
fp3 = Path("Datasets/Raw") / "hotels.csv"
datasets.append(Hotel(fp3))

# for dataset in datasets:
#     X = dataset.x_train
#     y = dataset.y_train.ravel()   
#     cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#     model = TPOTRegressor(generations=50, population_size=50, scoring='neg_root_mean_squared_error',
#                         cv=cv, verbosity=3, random_state=1, n_jobs=-1)
#     model.fit(X, y)
#     print("MODEL SCORE", model.score(dataset.x_test, dataset.y_test))
#     model.export(f'Models/tpot_{dataset.get_name()}_best_model.py')
# Wine: MODEL SCORE -0.5907459172139811
# Conrete: MODEL SCORE -5.432604661167217
# Hotels: MODEL SCORE -1.0863806416605724


# ----- SCIKIT SOLUTION -----
for dataset in datasets:
    X = dataset.x_train
    y = dataset.y_train.ravel()
    automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    resampling_strategy = 'cv',
    resampling_strategy_arguments={'folds': 5},
    metric=autosklearn.metrics.mean_squared_error,
    tmp_folder='Models_sk',
)
    automl.fit(X, y, dataset_name=dataset.get_name())
    predictions = automl.predict(dataset.x_test)
    print("Score reached:", sklearn.metrics.mean_squared_error(
        dataset.y_test, predictions))
    print(automl.leaderboard())
    print(automl.show_models())