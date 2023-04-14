#%%
import trackers
import transformation_functions
import os
import numpy as np
import scipy

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
Tracker_55 = trackers.Tracker(0, './Data/Trackers/ZF_DIP.csv', ctname="Slicer3D/Tracker55.mrk.json")
Tracker_M4_gross = trackers.Tracker(0, './Data/Trackers/ZF_MCP.csv', ctname="Slicer3D/Tracker_M4_gross.mrk.json")
Tracker_53 = trackers.Tracker(0, './Data/Trackers/DAU_DIP.csv', ctname="Slicer3D/Tracker53.mrk.json")
Tracker_M4_klein = trackers.Tracker(0, './Data/Trackers/DAU_MCP.csv', ctname="Slicer3D/Tracker_M4_klein.mrk.json")
Tracker_FT = trackers.Tracker(0, './Data/Trackers/FT.csv', ctname=None)
Tracker_52 = trackers.Tracker(0, './Data/Trackers/ZF_DIP.csv', ctname=None) # Basis, hinten an Fixteur externe

print(Tracker_52.t_ct_def)

# opti_taj_Marker data transformation in CT coordinatesystem
ct_traj_Tracker_52 = opti_traj_Tracker_52 * Tracker_52.t_ct_tr

# Winkel DAU

# Winkel ZF



#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]