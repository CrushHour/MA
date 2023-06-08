#%% Import
import sys
sys.path.append('./mujoco')
import transformation_functions as tf
import Konzepte.trackers as trackers
import numpy as np
from tqdm import tqdm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import stl
import my_write_parameters as mwp
import my_model as mwj
import yaml
from pyquaternion import Quaternion
import importlib
importlib.reload(tf)
importlib.reload(trackers)
from scipy.spatial.transform import Rotation as Rot

# %%Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

''' Laden des Testfiles als csv, opttrack Rohdaten '''
# %%Tracker
#Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
#[X	        Y	        Z	        W]	        [X	        Y	        Z]	
#opt_traj_55 = tf.csv_test_load(opt_data,"55")
#opt_traj_Tracker_52 = tf.csv_test_load(opt_data,"Tracker_52")
#opt_traj_Tracker_53 = tf.csv_test_load(opt_data,"Tracker 53")
#opt_traj_FTTracker = tf.csv_test_load(opt_data,"FT-Tracker-4")
#opt_traj_M4_gross = tf.csv_test_load(opt_data,"M4_gross")
#opt_traj_M4_klein = tf.csv_test_load(opt_data,"M4_klein")

# opt_traj_Marker_ZF_proximal = tf.marker_variable_id_linewise(opt_data,"Unlabeled 2023")
# opt_traj_Marker_DAU = tf.marker_variable_id_linewise(opt_data,"Unlabeled 2016")

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',test_path=test_metadata['path'])
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])
#Tracker_FT = tf.tracker_bone('FT',test_path=test_metadata['path'])
Basetracker = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path']) # Basis, hinten an Fixteur externe existiert nicht im CT

Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])
Marker_ZF_proximal = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])


# %% Berechnung der Markerpositionen im CT
i = 0

def construct_marker_rot(opt_info, ct_info):
    '''Do kabsch with points from different sources. Points must be in the same order.'''
    #print('Opt info: ', opt_info)
    #print('CT info: ', ct_info)
    T = Tracker_DAU_DIP.calculate_transformation_matrix(opt_info,ct_info)
    T = Tracker_DAU_DIP.invert_T(T)
    return T

for t in tqdm(range(len(Marker_DAU.opt_marker_trace))):
    Marker_DAU.T_opt_ct[t] = construct_marker_rot([Marker_DAU.opt_marker_trace[t],Tracker_DAU_DIP.T_proxi_opt[t,:3,3],Tracker_DAU_MCP.T_dist_opt[t,:3,3]],\
                                                  [Marker_DAU.marker_pos_ct[0], Tracker_DAU_DIP.t_proxi_CT, Tracker_DAU_MCP.t_dist_CT])
    
    Marker_ZF_proximal.T_opt_ct[t] = construct_marker_rot([Marker_ZF_proximal.opt_marker_trace[t],Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3]], \
                                                          [Marker_ZF_proximal.marker_pos_ct[0], Tracker_ZF_DIP.t_proxi_innen_CT, Tracker_ZF_DIP.t_proxi_aussen_CT])

# %% Build mujoco parameters
parameters = {'zf': dict(), 'dau': dict()}

parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_DIP.T_opt_ct[i,:3,:3]), Tracker_ZF_DIP.T_opt_ct[i,:3,3]])
parameters['zf']['pip'] = mwp.build_parameters([Quaternion(matrix=Marker_ZF_proximal.T_opt_ct[i,:3,:3]), Marker_ZF_proximal.T_opt_ct[i,:3,3]])
parameters['zf']['pip_marker'] = mwp.build_parameters([Quaternion(matrix=Marker_ZF_proximal.T_opt_ct[i,:3,:3]), Marker_ZF_proximal.T_opt_ct[i,:3,3]]) # blue
#parameters['zf']['mcp'] = mwp.build_parameters([Tracker_ZF_MCP.cog_rot_CT[i] ,Tracker_ZF_DIP.T_def_ct[i]])
parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[i,:3,:3]), Tracker_ZF_midhand.T_opt_ct[i,:3,3]])

parameters['zf']['dip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_DIP.T_proxi_aussen_opt[i,:3,3]]) # green
parameters['zf']['dip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_DIP.T_proxi_innen_opt[i,:3,3]]) # red
parameters['zf']['mcp_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_midhand.T_dist_aussen_opt[i,:3,3]]) # yellow
parameters['zf']['mcp_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_midhand.T_dist_innen_opt[i,:3,3]]) # white
parameters['zf']['pip_marker'] = mwp.build_parameters([[1,0,0,0], Marker_ZF_proximal.opt_marker_trace[i]]) # white


parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[i,:3,:3]), Tracker_DAU_DIP.T_opt_ct[i,:3,3]])
parameters['dau']['pip'] = mwp.build_parameters([Quaternion(matrix=Marker_DAU.T_opt_ct[i,:3,:3]), Marker_DAU.T_opt_ct[i,:3,3]])
parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[i,:3,:3]), Tracker_DAU_MCP.T_opt_ct[i,:3,3]])

parameters['dau']['pip_joint' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_DIP.T_proxi_opt[i,:3,3]]) # green
parameters['dau']['mcp_joint' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_MCP.T_dist_opt[i,:3,3]]) # yellow

with open("./mujoco/generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template.xml", "./mujoco/generated_parameters.yaml")
print("Model updated!")

# %% Make checks for plauability

DAU_COGs = tf.get_single_joint_file("./Data/Slicer3D/DAU_COG.mrk.json")

print('Diff DAU DIP:', DAU_COGs[0]-Tracker_DAU_DIP.cog_stl)
print('Diff DAU MCP:', DAU_COGs[2]-Tracker_DAU_MCP.cog_stl)

# %% Berechnung der Winkel zwischen den Markern und den Trackern
# angle ZF joints

alpha = np.zeros(len(Tracker_ZF_DIP.T_def_ct))
beta = np.zeros(len(Tracker_ZF_DIP.T_def_ct))
gamma = np.zeros(len(Tracker_ZF_DIP.T_def_ct))
delta = np.zeros(len(Tracker_ZF_DIP.T_def_ct))
epsilon = np.zeros(len(Tracker_ZF_DIP.T_def_ct))

for i in range(len(Tracker_ZF_DIP.T_def_ct)):
    alpha = tf.angle_between(1,2)
    beta = tf.angle_between(2,3)
    gamma = tf.angle_between(3,4)
    #calculate angeles in DAU joints
    #delta[i] = tf.angle_between(np.subtract(Tracker_DAU_DIP.T_def_ct[i],Tracker_DAU_DIP.dist_traj_CT[i]),np.subtract(Marker_DAU.T_opt_ct[i,:3,3],Tracker_DAU_DIP.dist_traj_CT[i]))*180/np.pi
    #epsilon[i] = tf.angle_between(np.subtract(Tracker_DAU_MCP.T_def_ct[i],Tracker_DAU_MCP.proxi_traj_CT[i]),np.subtract(Marker_DAU.T_opt_ct[i,:3,3],Tracker_DAU_MCP.proxi_traj_CT[i]))*180/np.pi

# plotten von delta und epsilon
plt.plot(delta)
plt.plot(epsilon)
plt.legend(['delta (DAU DIP)','epsilon (DAU PIP)'])
plt.show()

# %% Visualisierung der Marker und Tracker
# calculate spheres
# diese Berechnung gibt die Länge eines Fingers zurück.
# es würde aber viel mehr sinn machen den Abstand zwischen den Joints
# aus den mkr.json files zu nehmen.
ZF_PIP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_PIP.stl")
minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_PIP)
d_ZF_DIP_PIP = np.linalg.norm([maxx-minx, maxy-miny, maxz-minz])

d_ZF_Tracker_PIP = Marker_ZF_proximal.d_dist_CT

ZF_MCP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_MCP.stl")
minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_MCP)
d_ZF_MCP_PIP = np.linalg.norm([maxx-minx, maxy-miny, maxz-minz])
ZF_Tracker_lst = []
DAU_Tracker_lst = []
name_lst = []

radius_lst = []

tf.plot_class(0,ZF_Tracker_lst,DAU_Tracker_lst,name_lst,radius_lst, save=False, show=True)

interact(tf.plot_class, i = widgets.IntSlider(min=0,max=len(Tracker_ZF_DIP.track_traj_opt)-1,step=1,value=0),
         Trackers1 = widgets.fixed(ZF_Tracker_lst), 
         Trackers2 = widgets.fixed(DAU_Tracker_lst),
         names = widgets.fixed(name_lst),
         radius = widgets.fixed(radius_lst),
         show = widgets.fixed(True))
#%%
# test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
# test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]