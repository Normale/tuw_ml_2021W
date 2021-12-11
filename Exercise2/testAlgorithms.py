from Exercise2.Datasets.Wine import Wine
from Exercise2.Datasets.Concrete import Concrete
from Exercise2.Datasets.Hotel import Hotel
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
import numpy as np

# Temporary solution due to discrepancies between IDEs
if "Exercise2" not in os.getcwd(): 
    if "Exercise2" in os.listdir(os.getcwd()):
        os.chdir("Exercise2")
    else: 
        raise NotImplementedError

"""---------------- INITIALIZE DATASET --------------------"""

# # if input("Do you wish to load a previous state? (Y/N)") == "Y":
if False:
    name = input("Which state do you want to load?")
    dataset = pickle.load(open('Datasets/Dump/'+name, 'rb'))
else:
    filepath = Path("Datasets/Raw") / "winequality-red.csv"
    fp2 = Path("Datasets/Raw") / "ConcreteStrengthData.csv"
    fp3 = Path("Datasets/Raw") / "hotels.csv"

    dataset = Wine(filepath)
    name = 'Wine'
    #
    # dataset = Concrete(fp2)
    # name = 'Concrete'
    #
    # dataset = Hotel(fp3)
    # name = 'Hotel'

    # method = 'EN'
    # method = 'RF'
    method = 'SVM'


"""------------------- PREDICT STATE ----------------------"""
#
# lr_prediction = dataset.getLRPrediction()
# en_prediction = dataset.getENPrediction()
# rf_prediction = dataset.getRFPrediction()
# svm_prediction = dataset.getSVMPrediction()

"""------------------- GRADIENT DESCENT ----------------------"""

# best_sol, all_sol, all_paths = dataset.full_search_EN()
# best_sol, all_sol, all_paths = dataset.full_search_RF()
best_sol, all_sol, all_paths = dataset.full_search_SVM()

print("FINISHED")
file = open(os.path.abspath('GD_Dump/{}_{}_sol'.format(name, method)), 'wb+')
paths_file = open(os.path.abspath('GD_Dump/{}_{}_paths'.format(name, method)), 'wb+')
pickle.dump(all_sol, file)
pickle.dump(all_paths, paths_file)


"""----------------------- PLOTS ---------------------------"""

# plt.plot(lr_prediction, label="lr")
# plt.plot(en_prediction, label="en")
# plt.plot(rf_prediction, label="rf")
# plt.plot(svm_prediction, label="svm")
# plt.legend()
# plt.show()
#
# avg = (lr_prediction+en_prediction+rf_prediction+svm_prediction)/4
# plt.plot(lr_prediction - avg, label="diff lr")
# plt.plot(en_prediction - avg, label="diff en")
# plt.plot(rf_prediction - avg, label="diff rf")
# plt.plot(svm_prediction - avg, label="diff svm")
#
# plt.legend()
# plt.show()


"""----------------------- SAVE STATE ----------------------"""

# if input("Do you wish to save the current state? (Y/N)") == "Y":
#     name = input("Under which name do you want to save the state?")
#     file = open(os.path.abspath('Datasets/Dump/'+name), 'wb+')
#     pickle.dump(dataset, file)
