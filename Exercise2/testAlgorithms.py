from Exercise2.Datasets.Purchase import Purchase
import matplotlib.pyplot as plt
import os
import pickle


"""---------------- INITIALIZE DATASET --------------------"""

if input("Do you wish to load a previous state? (Y/N)") == "Y":
    name = input("Which state do you want to load?")
    dataset = pickle.load(open('Datasets/Dump/'+name, 'rb'))
else:
    # TODO: implement all models at once (use dictionaries and keys)
    dataset = Purchase()


"""------------------- PREDICT STATE ----------------------"""

lr_prediction = dataset.getLRPrediction()
en_prediction = dataset.getENPrediction()
rf_prediction = dataset.getRFPrediction()
svm_prediction = dataset.getSVMPrediction()


"""----------------------- PLOTS ---------------------------"""

plt.plot(lr_prediction, label="lr")
plt.plot(en_prediction, label="en")
plt.plot(rf_prediction, label="rf")
plt.plot(svm_prediction, label="svm")
plt.legend()
plt.show()

avg = (lr_prediction+en_prediction+rf_prediction+svm_prediction)/4
plt.plot(lr_prediction - avg, label="diff lr")
plt.plot(en_prediction - avg, label="diff en")
plt.plot(rf_prediction - avg, label="diff rf")
plt.plot(svm_prediction - avg, label="diff svm")

plt.legend()
plt.show()


"""----------------------- SAVE STATE ----------------------"""

if input("Do you wish to save the current state? (Y/N)") == "Y":
    name = input("Under which name do you want to save the state?")
    file = open(os.path.abspath('Datasets/Dump/'+name), 'wb+')
    pickle.dump(dataset, file)
