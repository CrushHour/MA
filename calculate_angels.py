#%%
import sys
sys.path.append('./mujoco')
import trackers
import transformation_functions as tf
import numpy as np
import scipy
from tqdm import tqdm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import stl
import my_write_parameters as mwp
import my_model as mwj
import yaml

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
#opti_traj_55 = tf.csv_test_load(opti_data,"55")
#opti_traj_Tracker_52 = tf.csv_test_load(opti_data,"Tracker_52")
#opti_traj_Tracker_53 = tf.csv_test_load(opti_data,"Tracker 53")
#opti_traj_FTTracker = tf.csv_test_load(opti_data,"FT-Tracker-4")
#opti_traj_M4_gross = tf.csv_test_load(opti_data,"M4_gross")
#opti_traj_M4_klein = tf.csv_test_load(opti_data,"M4_klein")

# opti_traj_Marker_ZF_proximal = tf.marker_variable_id_linewise(opti_data,"Unlabeled 2403")
# opti_traj_Marker_DAU = tf.marker_variable_id_linewise(opti_data,"Unlabeled 2016")

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',test_path=test_metadata['path'])
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])




#Tracker_FT = tf.tracker_bone('FT',test_path=test_metadata['path'])
Basetracker = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path']) # Basis, hinten an Fixteur externe existiert nicht im CT

Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])
Marker_ZF_proximal = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])

# calculate spheres
ZF_PIP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_PIP.stl")
minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_PIP)
d_ZF_DIP_PIP = np.linalg.norm([maxx-minx,maxy-miny,maxz-minz])

d_ZF_Tracker_PIP = Marker_ZF_proximal.d_dist_CT

ZF_MCP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_MCP.stl")
minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_MCP)
d_ZF_MCP_PIP = np.linalg.norm([maxx-minx,maxy-miny,maxz-minz])
# %% Visualisierung der Marker und Tracker
ZF_Tracker_lst = [Tracker_ZF_DIP.cog_traj_CT, Tracker_ZF_DIP.dist_traj_CT, Marker_ZF_proximal.ct_marker_trace, Tracker_ZF_midhand.proxi_traj_CT, Tracker_ZF_midhand.cog_traj_CT]
DAU_Tracker_lst = [Tracker_DAU_DIP.cog_traj_CT, Tracker_DAU_DIP.dist_traj_CT, Marker_DAU.ct_marker_trace, Tracker_DAU_MCP.proxi_traj_CT,Tracker_DAU_MCP.cog_traj_CT]
name_lst = ["Tracker_ZF_DIP.cog_traj_CT",'Tracker_ZF_DIP.dist_traj_CT' ,"Marker_ZF_proximal.ct_marker_trace", "Tracker_ZF_midhand.proxi_traj_CT","Tracker_ZF_midhand.cog_traj_CT",
            "Tracker_DAU_DIP.cog_traj_CT", "Tracker_DAU_DIP.dist_traj_CT", "Marker_DAU.ct_marker_trace", "Tracker_DAU_MCP.proxi_traj_CT","Tracker_DAU_MCP.cog_traj_CT"]



radius_lst = [0, d_ZF_DIP_PIP, d_ZF_Tracker_PIP, d_ZF_MCP_PIP, 0, \
              0, 0, Marker_DAU.d_cog_CT, 0, 0]

tf.plot_class(0,ZF_Tracker_lst,DAU_Tracker_lst,name_lst,radius_lst, save=False, show=True)

interact(tf.plot_class, i = widgets.IntSlider(min=0,max=len(Tracker_ZF_DIP.track_traj_opti)-1,step=1,value=0),
         Trackers1 = widgets.fixed(ZF_Tracker_lst), 
         Trackers2 = widgets.fixed(DAU_Tracker_lst),
         names = widgets.fixed(name_lst),
         radius = widgets.fixed(radius_lst),
         show = widgets.fixed(True))

# %% Calculate points of the joints, which are not known yet.
# On the DAU all points are known.
# On the ZF points of the distal joint are not known.
# Therefore the points of the distal joint are calculated with the help of the proximal joint.

# calculate points of distal jointk

# %% Check if Markers are unique
for i in range(len(Marker_DAU.ct_marker_trace)):
    print(np.subtract(Marker_DAU.ct_marker_trace[i],Marker_ZF_proximal.ct_marker_trace[i]))


# %% Build mujoco parameters
i = 0
system = 'opti'

parameters = {'zf': dict(), 'dau': dict()}

if system == 'CT':
    parameters['zf']['dip'] = mwp.build_parameters([Tracker_ZF_DIP.cog_rot_CT[i], Tracker_ZF_DIP.cog_traj_CT[i]])
    parameters['zf']['pip'] = mwp.build_parameters([[1,0,0,0], Marker_ZF_proximal.ct_marker_trace[i] + Marker_ZF_proximal.t_cog_CT])
    #parameters['zf']['mcp'] = mwp.build_parameters([Tracker_ZF_MCP.cog_rot_CT[i] ,Tracker_ZF_DIP.cog_traj_CT[i]])
    parameters['zf']['midhand'] = mwp.build_parameters([Tracker_ZF_midhand.cog_rot_CT[i], Tracker_ZF_midhand.cog_traj_CT[i]])

    parameters['dau']['dip'] = mwp.build_parameters([Tracker_DAU_DIP.cog_rot_CT[i], Tracker_DAU_DIP.cog_traj_CT[i]])
    parameters['dau']['pip'] = mwp.build_parameters([Marker_DAU.ct_marker_trace[i] + Marker_DAU.t_cog_CT, [1,0,0,0]])
    parameters['dau']['mcp'] = mwp.build_parameters([Tracker_DAU_MCP.cog_rot_CT[i], Tracker_DAU_MCP.cog_traj_CT[i]])
    with open("./mujoco/generated_parameters.yaml", "w") as outfile:
        yaml.dump(parameters, outfile)

elif system == 'opti':
    # Test für Opti system.
    parameters['zf']['midhand'] = mwp.build_parameters([Tracker_ZF_midhand.track_traj_opti[i][:4] ,Tracker_ZF_midhand.track_traj_opti[i][4:7]])

    parameters['dau']['dip'] = mwp.build_parameters([Tracker_DAU_DIP.track_traj_opti[i][:4], Tracker_DAU_DIP.track_traj_opti[i][4:7]])
    parameters['dau']['pip'] = mwp.build_parameters([[1,0,0,0], Marker_DAU.opti_marker_trace[i] + Marker_DAU.t_cog_CT])
    parameters['dau']['mcp'] = mwp.build_parameters([Tracker_DAU_MCP.track_traj_opti[i][:4], Tracker_DAU_MCP.track_traj_opti[i][4:7]])
    with open("./mujoco/generated_parameters.yaml", "w") as outfile:
        yaml.dump(parameters, outfile)

else:
    print('invalid System.')

    model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template_simple.xml", "./mujoco/generated_parameters.yaml")
    print("Model updated!")

# %% Make checks for plauability

DAU_COGs = tf.get_signle_joint_file("./Data/Slicer3D/DAU_COG.mrk.json")

print('Diff DAU DIP:', DAU_COGs[0]-Tracker_DAU_DIP.cog_stl)
print('Diff DAU MCP:', DAU_COGs[2]-Tracker_DAU_MCP.cog_stl)

# %% Berechnung der Winkel zwischen den Markern und den Trackern
# angle ZF joints

alpha = np.zeros(len(Tracker_ZF_DIP.cog_traj_CT))
beta = np.zeros(len(Tracker_ZF_DIP.cog_traj_CT))
gamma = np.zeros(len(Tracker_ZF_DIP.cog_traj_CT))
delta = np.zeros(len(Tracker_ZF_DIP.cog_traj_CT))
epsilon = np.zeros(len(Tracker_ZF_DIP.cog_traj_CT))

for i in range(len(Tracker_ZF_DIP.cog_traj_CT)):
    alpha = tf.angle_between(1,2)
    beta = tf.angle_between(2,3)
    gamma = tf.angle_between(3,4)
    #calculate angeles in DAU joints
    delta[i] = tf.angle_between(np.subtract(Tracker_DAU_DIP.cog_traj_CT[i],Tracker_DAU_DIP.dist_traj_CT[i]),np.subtract(Marker_DAU.cog_traj_CT[i],Tracker_DAU_DIP.dist_traj_CT[i]))*180/np.pi
    epsilon[i] = tf.angle_between(np.subtract(Tracker_DAU_MCP.cog_traj_CT[i],Tracker_DAU_MCP.proxi_traj_CT[i]),np.subtract(Marker_DAU.cog_traj_CT[i],Tracker_DAU_MCP.proxi_traj_CT[i]))*180/np.pi

# plotten von delta und epsilon
plt.plot(delta)
plt.plot(epsilon)
plt.legend(['delta (DAU DIP)','epsilon (DAU PIP)'])
plt.show()

#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]