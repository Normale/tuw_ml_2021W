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

with open(os.path.abspath('GD_Dump/{}_{}_paths'.format(name, method)), 'rb+') as f:
    d_p = pickle.load(f)
with open(os.path.abspath('GD_Dump/{}_{}_sol'.format(name, method)), 'rb+') as f:
    d = pickle.load(f)

# x = []
# y = []
# z = []
#
# for pred in d:
#     print(pred)
#     x.append(pred[0]['alpha']['value'])
#     y.append(pred[0]['l1_ratio']['value'])
#     z.append(pred[1])
#
#
# # x = np.log10(x)
# # y = np.log10(y)
#
# # # Creating figure
# # fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
# ax.scatter(x, y, z)
# plt.show()


i=0
plt.figure(dpi=100)
# ax = plt.axes(projection="3d")

for pred in d_p:
    if i < 2:
        i+=1
        continue
    x = []
    y = []
    z = []

    for point in pred[0]:
        x.append(point[0])
        # y.append(point[1])
    for cost in pred[1]:
        z.append(cost)
        # x = np.log10(x)
        # y = np.log10(y)

    # # Creating figure
    # fig = plt.figure()
    # ax.plot(x, y, z)
    # ax.scatter(x, y, z, s=3, depthshade=False)
    plt.plot(x,z)
plt.xlabel('n_estimators')
plt.title('Gradient descent for RF: Wine')
plt.ylabel('cost')
# ax.set_ylabel('l1_ratio')
# ax.set_zlabel('cost')
# ax.set_zlim(0.38,0.44)
plt.xlim(0,120)
# plt.ylim(0,4)
plt.show()









