#%%
import trackers
import transformation_functions as tf
import numpy as np
import scipy
from tqdm import tqdm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opti_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'
#opti_data = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'

''' Laden des Testfiles als csv, Optitrack Rohdaten '''
# Tracker
#Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
#[X	        Y	        Z	        W]	        [X	        Y	        Z]	
opti_traj_55 = tf.csv_test_load(opti_data,"55")
opti_traj_Tracker_52 = tf.csv_test_load(opti_data,"Tracker_52")
opti_traj_Tracker_53 = tf.csv_test_load(opti_data,"Tracker 53")
opti_traj_FTTracker = tf.csv_test_load(opti_data,"FT-Tracker-4")
opti_traj_M4_gross = tf.csv_test_load(opti_data,"M4_gross")
opti_traj_M4_klein = tf.csv_test_load(opti_data,"M4_klein")

# Marker
# 
                        
# opti_traj_Marker_ZF_distal = tf.marker_variable_id_linewise(opti_data,"M4_klein")
# opti_traj_Marker_ZF_proximal = tf.marker_variable_id_linewise(opti_data,"Unlabeled 2403")
# opti_traj_Marker_DAU = tf.marker_variable_id_linewise(opti_data,"Unlabeled 2016")

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Marker_DAU = tf.marker_bone(finger_name='DAU_DIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])
Marker_ZF_proximal = tf.marker_bone(finger_name="ZF_DIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])

Tracker_ZF_DIP = tf.tracker_bone(0, './Data/Trackers/ZF_DIP.csv', ctname="Slicer3D/Tracker55.mrk.json")
Tracker_ZF_MCP = tf.tracker_bone(0, './Data/Trackers/ZF_MCP.csv', ctname="Slicer3D/Tracker_M4_gross.mrk.json")
Tracker_DAU_DIP = tf.tracker_bone(0, './Data/Trackers/DAU_DIP.csv', ctname="Slicer3D/Tracker53.mrk.json")
Tracker_DAU_MCP = tf.tracker_bone(0, './Data/Trackers/DAU_MCP.csv', ctname="Slicer3D/Tracker_M4_klein.mrk.json")
Tracker_FT = tf.tracker_bone(0, './Data/Trackers/FT.csv', ctname=None)
Basetracker = tf.tracker_bone(0, './Data/Trackers/ZF_DIP.csv', ctname=None) # Basis, hinten an Fixteur externe


"""# opti_taj_Marker data transformation in CT coordinatesystem
ct_traj_Marker_ZF_proximal = np.zeros((len(opti_traj_Marker_ZF_proximal),3))
ct_traj_Marker_DAU = np.zeros((len(opti_traj_Marker_DAU),3))
for i in range(len(opti_traj_Marker_ZF_proximal)):
    # Marker ZF proximal
    ct_traj_Marker_ZF_proximal[i] = np.matmul(Tracker_52.t_ct_def[:3,:3], opti_traj_Marker_ZF_proximal[i])
    ct_traj_Marker_ZF_proximal[i] = ct_traj_Marker_ZF_proximal[i] + Tracker_52.t_ct_def[:3,3]

    # Marker DIP
    ct_traj_Marker_DAU[i] = np.matmul(Tracker_52.t_ct_def[:3,:3], opti_traj_Marker_DAU[i])
    ct_traj_Marker_DAU[i] = ct_traj_Marker_DAU[i] + Tracker_52.t_ct_def[:3,3]"""

# %% Visualisierung der Marker und Tracker
interact(tf.plot_class, i = widgets.IntSlider(min=0,max=len(Tracker_ZF_DIP.track_traj_opti)-1,step=1,value=0),
         Tracker_ZF_DIP=fixed(Tracker_ZF_DIP.cog_traj_CT),
         Tracker_ZF_MCP=fixed(Tracker_ZF_MCP.cog_traj_CT),
         Tracker_DAU_DIP=fixed(Tracker_DAU_DIP.cog_traj_CT),
         Tracker_DAU_MCP=fixed(Tracker_DAU_MCP.cog_traj_CT),
         Tracker_FT=fixed(Tracker_FT.cog_traj_CT),Marker_ZF_proximal=fixed(Marker_ZF_proximal.cog_traj_CT),
         Marker_DAU=fixed(Marker_DAU.cog_traj_CT),
         Basetracker=fixed(Basetracker.cog_traj_CT))

# angle ZF joints
alpha = tf.angle_between(1,2)
beta = tf.angle_between(2,3)
gamma = tf.angle_between(3,4)
#calculate angeles in DAU joints
alpha = tf.angle_between(1,2)
beta = tf.angle_between(2,3)


#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]