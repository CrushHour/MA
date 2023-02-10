#%%
import trackers
import os

data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opti_data = 'Take 2023-01-31 06.11.42 PM.csv'

ZF_DIP_raw = trackers.read_markerdata(os.path.join(data_path,opti_data), '55', True)
Tracker_55 = trackers.Tracker(0, './Data/Trackers/DAU_DIP.csv')
#%%