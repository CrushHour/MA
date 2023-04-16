#%%
import trackers
import transformation_functions
import os
import numpy as np
import scipy
from tqdm import tqdm

# Definition der Pfade
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opti_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'
#opti_data = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'

''' Laden des Testfiles als csv, Optitrack Rohdaten '''
# Tracker
opti_traj_55 = transformation_functions.csv_test_load(opti_data,"55")
opti_traj_Tracker_52 = transformation_functions.csv_test_load(opti_data,"Tracker_52")
opti_traj_Tracker_53 = transformation_functions.csv_test_load(opti_data,"Tracker 53")
opti_traj_FTTracker = transformation_functions.csv_test_load(opti_data,"FT-Tracker-4")
opti_traj_M4_gross = transformation_functions.csv_test_load(opti_data,"M4_gross")
opti_traj_M4_klein = transformation_functions.csv_test_load(opti_data,"M4_klein")

# Marker
# abgebrochen
#opti_traj_Marker_ZF_distal = transformation_functions.marker_variable_id_linewise(opti_data,"M4_klein")
opti_traj_Marker_ZF_proximal = transformation_functions.marker_variable_id_linewise(opti_data,"Unlabeled 2403")
opti_traj_Marker_DAU = transformation_functions.marker_variable_id_linewise(opti_data,"Unlabeled 2016")
Marker_DAU = transformation_functions.bone_stl(finger_name='DAU_DIP')
Marker_ZF_proximal = transformation_functions.bone_stl(finger_name="ZF_DIP")
# Laden des Testfiles als json vom Stream.

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
# csv tracker entries are: 0: id, 1: name, 2: x, 3: y, 4: z, 5: qx, 6: qy, 7: qz, 8: qw?
#Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
#X	Y	Z	W	X	Y	Z	

Tracker_55 = trackers.Tracker(0, './Data/Trackers/ZF_DIP.csv', ctname="Slicer3D/Tracker55.mrk.json")
Tracker_M4_gross = trackers.Tracker(0, './Data/Trackers/ZF_MCP.csv', ctname="Slicer3D/Tracker_M4_gross.mrk.json")
Tracker_53 = trackers.Tracker(0, './Data/Trackers/DAU_DIP.csv', ctname="Slicer3D/Tracker53.mrk.json")
Tracker_M4_klein = trackers.Tracker(0, './Data/Trackers/DAU_MCP.csv', ctname="Slicer3D/Tracker_M4_klein.mrk.json")
Tracker_FT = trackers.Tracker(0, './Data/Trackers/FT.csv', ctname=None)
Tracker_52 = trackers.Tracker(0, './Data/Trackers/ZF_DIP.csv', ctname=None) # Basis, hinten an Fixteur externe

print(Tracker_52.t_ct_def)

# opti_taj_Marker data transformation in CT coordinatesystem
ct_traj_Marker_ZF_proximal = np.zeros((len(opti_traj_Marker_ZF_proximal),3))
ct_traj_Marker_DAU = np.zeros((len(opti_traj_Marker_DAU),3))
for i in range(len(opti_traj_Marker_ZF_proximal)):
    # Marker ZF proximal
    ct_traj_Marker_ZF_proximal[i] = np.matmul(Tracker_52.t_ct_def[:3,:3], opti_traj_Marker_ZF_proximal[i])
    ct_traj_Marker_ZF_proximal[i] = ct_traj_Marker_ZF_proximal[i] + Tracker_52.t_ct_def[:3,3]

    # Marker DIP
    ct_traj_Marker_DAU[i] = np.matmul(Tracker_52.t_ct_def[:3,:3], opti_traj_Marker_DAU[i])
    ct_traj_Marker_DAU[i] = ct_traj_Marker_DAU[i] + Tracker_52.t_ct_def[:3,3]
print(ct_traj_Marker_ZF_proximal)

# postitions of helper_points 2 is derived from tracker 55
# r_rel_hp2 is the relative position of helper point 2 and is the difference of the position of the helper point 2 and the position of the tracker 52
hp_DAU_DIP = np.zeros((len(opti_traj_55),3))
hp_DAU_MCP = np.zeros((len(opti_traj_M4_klein),3))
r_rel_hp2 = np.zeros((3,3))
r_rel_hp4 = np.zeros((3,3))
r_rel_hp2 = np.subtract(hp_DAU_DIP, Tracker_55.t_ct_def[:3,3])
r_rel_hp4 = np.subtract(hp_DAU_MCP, Tracker_M4_klein.t_ct_def[:3,3])
# calculate relative position of helper points
for i in range(len(opti_traj_55)):
    # helper point 2
    hp_DAU_DIP[i] = np.matmul(Tracker_55.t_ct_def[:3,:3], opti_traj_55[i])
    hp_DAU_DIP[i] = hp_DAU_DIP[i] + r_rel_hp2 * Tracker_55.t_ct_def[:3,:3]
    # helper point 4
    hp_DAU_MCP[i] = np.matmul(Tracker_M4_klein.t_ct_def[:3,:3], opti_traj_M4_klein[i])
    hp_DAU_MCP[i] = hp_DAU_MCP[i] + r_rel_hp4 * Tracker_M4_klein.t_ct_def[:3,:3]

# angle ZF joints



#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]