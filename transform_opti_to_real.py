#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
# %%
path_csv = './Optitrack/Take 2022-12-13 01.59.35 PM.csv'
data = pd.read_csv(path_csv, header=2)
num_trackers = 1

#%% transformation optitrack tracker to real tracker
path_csv = './Optitrack/Tracker Nico.csv'
for i in range(num_trackers):
    print(i)

#%% transformation tracker to bone
print('hello world!')