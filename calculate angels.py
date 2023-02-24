#%%
import trackers
import transformation_functions
import os
import numpy as np
import scipy

# Definition der Pfade
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opti_data = 'Take 2023-01-31 06.11.42 PM.csv'

# Laden des Testfiles als csv

# Laden des Testfiles als json vom Stream.

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_55 = trackers.Tracker(0, './Data/Trackers/ZF_DIP.csv', ctname="Slicer3D/Tracker55.mrk.json")
Tracker_M4_gross = trackers.Tracker(0, './Data/Trackers/ZF_MCP.csv', ctname="Slicer3D/Tracker_M4_gross.mrk.json")
Tracker_53 = trackers.Tracker(0, './Data/Trackers/DAU_DIP.csv', ctname="Slicer3D/Tracker53.mrk.json")
Tracker_M4_klein = trackers.Tracker(0, './Data/Trackers/DAU_MCP.csv', ctname="Slicer3D/Tracker_M4_klein.mrk.json")
Tracker_FT = trackers.Tracker(0, './Data/Trackers/FT.csv', ctname=None) # Tracker_52

# Winkel ZF



#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]