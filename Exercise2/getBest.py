import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pickle
from pathlib import Path
import numpy as np
import time


name = 'Wine'
# name = 'Concrete'
# name = 'Hotel'

# method = 'EN'
# method = 'SVM'
method = 'RF'

with open(os.path.abspath('GD_Dump/RF_Housing_paths'), 'rb+') as f:
    d_p = pickle.load(f)
# with open(os.path.abspath('GD_Dump/{}_{}_paths'.format(name, method)), 'rb+') as f:
#     d_p = pickle.load(f)
# with open(os.path.abspath('GD_Dump/{}_{}_sol'.format(name, method)), 'rb+') as f:
#     d = pickle.load(f)

i=0
best = (1,1000)
for pred in d_p:
    if i < 2:
        i += 1
        continue

    params = pred[0]
    costs = pred[1]
    for i in range(len(params)):
        if costs[i] < best[1]:
            print("--------------------------------")
            print(best)
            best = params[i], costs[i]
            print(best)


print(best)
