import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pickle
from pathlib import Path
import numpy as np


with open(os.path.abspath('Datasets/Dump/EN'), 'rb+') as f:
    d = pickle.load(f)

x = []
y = []
z = []

for pred in d:
    print(pred)
    x.append(pred[0]['alpha']['value'])
    y.append(pred[0]['l1_ratio']['value'])
    z.append(pred[1])


# x = np.log10(x)
# y = np.log10(y)

# # Creating figure
# fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

ax.scatter3D(x, y, z)
print(x)
print(y)
print(z)
plt.show()
# plt.plot(z)
# plt.show()




