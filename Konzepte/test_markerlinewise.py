import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt

# Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'


Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])
