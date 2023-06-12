#%%
import numpy as np
import os
import sys
os.chdir(r'C:/GitHub/MA')
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt

# Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

try:
    save_name = './Data/' + test_metadata['marker_IDs'][1] + '_opt_marker_trace.npy'
    os.remove(save_name)
except:
    print('No file to delete', test_metadata['marker_IDs'][1])
try:
    save_name = './Data/' + test_metadata['marker_IDs'][0] + '_opt_marker_trace.npy'
    os.remove(save_name)
except:
    print('No file to delete', test_metadata['marker_IDs'][0])

#%%
#Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])
#Marker_ZF_intermedial = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])

# %%
#path = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'
path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

#raw_data = csv_test_load(path, '55')
marker_ID = 'Unlabeled 2016'
#marker_ID = 'Unlabeled 2403'
marker_data = tf.marker_variable_id_linewise(path, marker_ID, "csv")
plt.plot(marker_data)
plt.show()

# %%
