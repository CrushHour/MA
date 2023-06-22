#%% Import
import sys
sys.path.append('./mujoco')
import transformation_functions as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import my_write_parameters as mwp
import my_model as mwj
import yaml
from pyquaternion import Quaternion
import importlib
importlib.reload(tf)
#from scipy.spatial.transform import Rotation as Rot

# %%Definition der Pfade
hand_metadata = tf.get_json('hand_metadata.json')
test_number = 1
if test_number == 1:
    test_metadata = tf.get_json('test_metadata.json')['Take 2023-01-31 06.11.42 PM.csv']
elif test_number == 0:
    test_metadata = tf.get_test_metadata('test_metadata.json')['optitrack-20230130-234800.json']
else:
    test_metadata = tf.get_test_metadata('test_metadata.json')['2023_01_31_18_12_48.json']

# %%Tracker
#Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
#[X	        Y	        Z	        W]	        [X	        Y	        Z]

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',test_path=test_metadata['path'])
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])
#Tracker_FT = tf.tracker_bone('FT',test_path=test_metadata['path'])

Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][1])
Marker_ZF_intermedial = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'], init_marker_ID=test_metadata['marker_IDs'][0])
ZF_MCP = tf.marker_bone(finger_name="ZF_MCP",test_path=test_metadata['path'], init_marker_ID='')

# %% Berechnung der Markerpositionen im CT

def construct_marker_rot(opt_info, ct_info):
    '''Do kabsch with points from different sources. Points must be in the same order.'''
    T = Tracker_DAU_DIP.kabsch(ct_info, opt_info)
    return T

for t in tqdm(range(len(Marker_DAU.opt_marker_trace))):
    Marker_DAU.T_opt_ct[t] = construct_marker_rot([Tracker_DAU_DIP.T_proxi_innen_opt[t,:3,3], Tracker_DAU_DIP.T_proxi_aussen_opt[t,:3,3],Tracker_DAU_MCP.T_dist_innen_opt[t,:3,3],Tracker_DAU_MCP.T_dist_aussen_opt[t,:3,3]],\
                                                  [Tracker_DAU_DIP.T_proxi_innen_CT[:3,3], Tracker_DAU_DIP.T_proxi_aussen_CT[:3,3],Tracker_DAU_MCP.T_dist_innen_CT[:3,3],Tracker_DAU_MCP.T_dist_aussen_CT[:3,3]])
    Marker_DAU.update_joints(t)

    #Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot([Marker_ZF_intermedial.opt_marker_trace[t],Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3]], \
    #                                                      [np.array(Marker_ZF_intermedial.marker_pos_ct[0]), Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3]])
    Marker_ZF_intermedial.T_opt_ct[t] = Tracker_ZF_DIP.T_opt_ct[t]

    # update loop auf basis aller bekannten Punkte
    for j in range(3):
        Marker_ZF_intermedial.update_joints(t)

        ZF_MCP.T_opt_ct[t] = construct_marker_rot([Tracker_ZF_midhand.T_dist_innen_opt[t,:3,3],Tracker_ZF_midhand.T_dist_aussen_opt[t,:3,3],Marker_ZF_intermedial.T_proxi_opt[t,0,:3,3],Marker_ZF_intermedial.T_proxi_opt[t,1,:3,3]], \
                                                [Tracker_ZF_midhand.T_dist_innen_CT[:3,3],Tracker_ZF_midhand.T_dist_aussen_CT[:3,3],Marker_ZF_intermedial.T_proxi_CT[0,:3,3], Marker_ZF_intermedial.T_proxi_CT[1,:3,3]])
        ZF_MCP.update_joints(t)

        Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot([Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3], ZF_MCP.T_dist_opt[t,0,:3,3], ZF_MCP.T_dist_opt[t,1,:3,3]], \
                                                          [Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3], ZF_MCP.T_dist_CT[0,:3,3], ZF_MCP.T_dist_CT[1,:3,3]])


# %% Build mujoco parameters
#i = 4553
i = 5831

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
parameters['dau']['mcp_joint2' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_MCP.T_proxi_opt[i,:3,3]])
parameters['dau']['pip_marker'] = mwp.build_parameters([[1,0,0,0], Marker_DAU.opt_marker_trace[i]])
parameters['dau']['mcp_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_MCP.T_proxi_aussen_opt[i,:3,3]]) 
parameters['dau']['mcp_joint_innen' ] = mwp.build_parameters([[1,0,0,0], Tracker_DAU_MCP.T_proxi_innen_opt[i,:3,3]]) 


with open("./mujoco/generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template.xml", "./mujoco/generated_parameters.yaml")
print("Model updated!")

# %% plotten von delta und epsilon


if __name__=="__main__":
    alpha = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    beta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    gamma = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    delta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    epsilon = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    zeta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    eta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    theta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    ita = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    ita2 = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    q = []

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

        zeta[i] = tf.angle_between(Tracker_DAU_MCP.v_opt[i,:2],Tracker_DAU_MCP.v_opt[0,:2])*180/np.pi
        eta[i] = tf.angle_between([Tracker_DAU_MCP.v_opt[i,0],Tracker_DAU_MCP.v_opt[i,2]],[Tracker_DAU_MCP.v_opt[0,0],Tracker_DAU_MCP.v_opt[0,2]])*180/np.pi
        
        axis=np.subtract(Tracker_DAU_MCP.T_proxi_innen_opt[i,:3,3],Tracker_DAU_MCP.T_proxi_aussen_opt[i,:3,3])
        axisp = np.cross(axis,Tracker_DAU_MCP.v_opt[i])
        ita2[i] = tf.angle_axis(v1=Tracker_DAU_MCP.v_opt[i],v2=Tracker_DAU_MCP.v_opt[0],axis=axisp)
        ita[i] = tf.angle_between(v1=Tracker_DAU_MCP.v_opt[i],v2=Tracker_DAU_MCP.v_opt[0])*180/np.pi
    alpha = tf.interpolate_1d(alpha)
    alpha = Tracker_DAU_DIP.delete_outliers(alpha)
    alpha = tf.interpolate_1d(alpha)
    beta = tf.interpolate_1d(beta)
    beta = Tracker_DAU_DIP.delete_outliers(beta, 1.5)
    beta = tf.interpolate_1d(beta)
    gamma = tf.interpolate_1d(gamma)
    gamma = Tracker_DAU_DIP.delete_outliers(gamma)
    gamma = tf.interpolate_1d(gamma)
    delta = tf.interpolate_1d(delta)
    delta = Tracker_DAU_DIP.delete_outliers_local(delta, 2.5, 1000)
    delta = tf.interpolate_1d(delta)
    epsilon = tf.interpolate_1d(epsilon)
    epsilon = Tracker_DAU_DIP.delete_outliers_local(epsilon, 1.5, 1000)
    epsilon = tf.interpolate_1d(epsilon)
    #tf.plot_angels([delta, epsilon], ['delta (DIP)','epsilon (PIP)'], 'Angles in Thumb joints', save_plots=True)
    #tf.plot_angels([alpha, beta, gamma], ['alpha (DIP)', 'beta (PIP)', 'gamma (MCP)'], 'Angles in Index finger joints', save_plots=True)
    tf.plot_angels([ita, ita2], ['around joint', 'complexe version'], 'Angles in Thumb MCP joint to midhand', save_plots=False)
    #tf.plot_quaternion(q, 'Quaternion Tracker', save_plots=True)
# %%