#%% Import
import sys
sys.path.append('./mujoco')
import transformation_functions as tf
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
from scipy.spatial.transform import Rotation as Rot

# %%Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

'''  '''
# %%Tracker
#Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
#[X	        Y	        Z	        W]	        [X	        Y	        Z]

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',test_path=test_metadata['path'])
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])
#Tracker_FT = tf.tracker_bone('FT',test_path=test_metadata['path'])
Basetracker = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path']) # Basis, hinten an Fixteur externe existiert nicht im CT

Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])
Marker_ZF_intermedial = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])
ZF_MCP = tf.marker_bone(finger_name="ZF_MCP",test_path=test_metadata['path'], init_marker_ID='')

# %% Berechnung der Markerpositionen im CT

def construct_marker_rot(opt_info, ct_info):
    '''Do kabsch with points from different sources. Points must be in the same order.'''
    T = Tracker_DAU_DIP.kabsch(ct_info, opt_info)
    return T

for t in tqdm(range(len(Marker_DAU.opt_marker_trace))):
    Marker_DAU.T_opt_ct[t] = construct_marker_rot([Marker_DAU.opt_marker_trace[t],Tracker_DAU_DIP.T_proxi_innen_opt[t,:3,3], Tracker_DAU_DIP.T_proxi_aussen_opt[t,:3,3],Tracker_DAU_MCP.T_dist_innen_opt[t,:3,3],Tracker_DAU_MCP.T_dist_aussen_opt[t,:3,3]],\
                                                  [Marker_DAU.marker_pos_ct[0], Tracker_DAU_DIP.T_proxi_innen_CT[:3,3], Tracker_DAU_DIP.T_proxi_aussen_CT[:3,3],Tracker_DAU_MCP.T_dist_innen_CT[:3,3],Tracker_DAU_MCP.T_dist_aussen_CT[:3,3]])
    Marker_DAU.update_joints(t)

    Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot([Marker_ZF_intermedial.opt_marker_trace[t],Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3]], \
                                                          [np.array(Marker_ZF_intermedial.marker_pos_ct[0]), Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3]])
    
    # update loop auf basis aller bekannten Punkte
    for j in range(5):
        Marker_ZF_intermedial.update_joints(t)

        ZF_MCP.T_opt_ct[t] = construct_marker_rot([Tracker_ZF_midhand.T_dist_innen_opt[t,:3,3],Tracker_ZF_midhand.T_dist_aussen_opt[t,:3,3],Marker_ZF_intermedial.T_proxi_opt[t,0,:3,3],Marker_ZF_intermedial.T_proxi_opt[t,1,:3,3]], \
                                                [Tracker_ZF_midhand.T_dist_innen_CT[:3,3],Tracker_ZF_midhand.T_dist_aussen_CT[:3,3],Marker_ZF_intermedial.T_proxi_CT[0,:3,3], Marker_ZF_intermedial.T_proxi_CT[1,:3,3]])
        ZF_MCP.update_joints(t)

        Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot([Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3], ZF_MCP.T_dist_opt[t,0,:3,3], ZF_MCP.T_dist_opt[t,1,:3,3]], \
                                                          [Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3], ZF_MCP.T_dist_CT[0,:3,3], ZF_MCP.T_dist_CT[1,:3,3]])


# %% Build mujoco parameters
i = 4553

parameters = {'zf': dict(), 'dau': dict()}

# STL
parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_DIP.T_opt_ct[i,:3,:3]), Tracker_ZF_DIP.T_opt_ct[i,:3,3]])
parameters['zf']['pip'] = mwp.build_parameters([Quaternion(matrix=Marker_ZF_intermedial.T_opt_ct[i,:3,:3]), Marker_ZF_intermedial.T_opt_ct[i,:3,3]])
parameters['zf']['mcp'] = mwp.build_parameters([Quaternion(matrix=ZF_MCP.T_opt_ct[i,:3,:3]) ,ZF_MCP.T_opt_ct[i,:3,3]])
parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[i,:3,:3]), Tracker_ZF_midhand.T_opt_ct[i,:3,3]])

# green, red, yellow, white, blue balls
parameters['zf']['dip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_DIP.T_proxi_aussen_opt[i,:3,3]]) 
parameters['zf']['dip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_DIP.T_proxi_innen_opt[i,:3,3]]) 
parameters['zf']['pip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Marker_ZF_intermedial.T_proxi_opt[i,1,:3,3]]) 
parameters['zf']['pip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Marker_ZF_intermedial.T_proxi_opt[i,0,:3,3]]) 
parameters['zf']['mcp_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_midhand.T_dist_aussen_opt[i,:3,3]]) 
parameters['zf']['mcp_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Tracker_ZF_midhand.T_dist_innen_opt[i,:3,3]]) 
parameters['zf']['pip_marker'] = mwp.build_parameters([[1,0,0,0], Marker_ZF_intermedial.opt_marker_trace[i]]) 

# STL
parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[i,:3,:3]), Tracker_DAU_DIP.T_opt_ct[i,:3,3]])
parameters['dau']['pip'] = mwp.build_parameters([Quaternion(matrix=Marker_DAU.T_opt_ct[i,:3,:3]), Marker_DAU.T_opt_ct[i,:3,3]])
parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[i,:3,:3]), Tracker_DAU_MCP.T_opt_ct[i,:3,3]])

# green, yellow balls
parameters['dau']['pip_joint' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_DIP.T_proxi_opt[i,:3,3]]) 
parameters['dau']['mcp_joint' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_MCP.T_dist_opt[i,:3,3]]) 
parameters['dau']['pip_marker'] = mwp.build_parameters([[1,0,0,0], Marker_DAU.opt_marker_trace[i]])


with open("./mujoco/generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template.xml", "./mujoco/generated_parameters.yaml")
print("Model updated!")

# %% Berechnung der Winkel zwischen den Markern und den Trackern
# angle ZF joints

alpha = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
beta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
gamma = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
delta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
epsilon = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))

vZF_PIP = np.subtract(np.mean(Marker_ZF_intermedial.T_dist_opt[i,:,:3,3],axis=0),np.mean(Marker_ZF_intermedial.T_proxi_opt[i,:,:3,3],axis=0))
vZF_MCP = np.subtract(np.mean(ZF_MCP.T_dist_opt[i,:,:3,3],axis=0),np.mean(ZF_MCP.T_proxi_opt[i,:,:3,3],axis=0))


for i in range(len(Marker_ZF_intermedial.opt_marker_trace)):
    vZF_PIP = np.subtract(np.mean(Marker_ZF_intermedial.T_dist_opt[i,:,:3,3],axis=0),np.mean(Marker_ZF_intermedial.T_proxi_opt[i,:,:3,3],axis=0))
    vZF_MCP = np.subtract(np.mean(ZF_MCP.T_dist_opt[i,:,:3,3],axis=0),np.mean(ZF_MCP.T_proxi_opt[i,:,:3,3],axis=0))

    alpha[i] = tf.angle_between(np.subtract(Tracker_ZF_DIP.T_dist_opt[i,:3,3],Tracker_ZF_DIP.T_proxi_opt[i,:3,3]),vZF_PIP)*180/np.pi
    beta[i] = tf.angle_between(vZF_PIP,vZF_MCP)*180/np.pi
    gamma[i] = tf.angle_between(vZF_MCP,Tracker_ZF_midhand.v_opt[i])*180/np.pi
    #calculate angeles in DAU joints
    delta[i] = tf.angle_between(np.subtract(np.mean(Marker_DAU.T_dist_opt[i,:,:3,3],axis=0),np.mean(Marker_DAU.T_proxi_opt[i,:,:3,3],axis=0)),np.subtract(Tracker_DAU_DIP.T_dist_opt[i,:3,3],Tracker_DAU_DIP.T_proxi_opt[i,:3,3]))*180/np.pi
    epsilon[i] = tf.angle_between(np.subtract(Tracker_DAU_MCP.T_dist_opt[i,:3,3],Tracker_DAU_MCP.T_proxi_opt[i,:3,3]),np.subtract(np.mean(Marker_DAU.T_dist_opt[i,:,:3,3],axis=0),np.mean(Marker_DAU.T_proxi_opt[i,:,:3,3],axis=0)))*180/np.pi

# plotten von delta und epsilon
plt.plot(delta)
plt.plot(epsilon)
plt.legend(['delta (DAU DIP)','epsilon (DAU PIP)'])
plt.ylabel('[°]')
plt.title('Angles in Thumb joints')
plt.show()
plt.close()

# plotten von alpha, beta, gamma
plt.plot(alpha)
plt.plot(beta)
plt.plot(gamma)
plt.legend(['alpha (DIP)', 'beta (PIP)', 'gamma (MCP)'])
plt.ylabel('[°]')
plt.title('Angles in Index finger joints')
plt.show()
plt.close()

# %% Visualisierung der Marker und Tracker
# calculate spheres
# diese Berechnung gibt die Länge eines Fingers zurück.
# es würde aber viel mehr sinn machen den Abstand zwischen den Joints
# aus den mkr.json files zu nehmen.
ZF_PIP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_PIP.stl")
minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_PIP)
d_ZF_DIP_PIP = np.linalg.norm([maxx-minx, maxy-miny, maxz-minz])

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