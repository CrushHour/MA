#%%
import trackers
import transformation_functions
import os
import numpy as np
import scipy

data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opti_data = 'Take 2023-01-31 06.11.42 PM.csv'

ZF_DIP_raw = trackers.read_markerdata(os.path.join(data_path,opti_data), '55', True)
Tracker_55_def = trackers.Tracker(0, './Data/Trackers/DAU_DIP.csv')
Tracker_55_CT = 




#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]