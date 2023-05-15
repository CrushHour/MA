#%% Import packages
import sys
sys.path.append('./mujoco')
import transformation_functions as tf
import trackers
import my_write_parameters as mwp
import my_model as mwj
import yaml
from pyquaternion import Quaternion
import importlib
importlib.reload(tf)
importlib.reload(trackers)

#%% Definition der Pfade
test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
hand_metadata = tf.get_json('hand_metadata.json')
data_path = 'Data/test_01_31/'
test_file = '2023_01_31_18_12_48.json'
opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

#%% Laden der Klassen
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])

#%% Erstellen den Mujocu Modells
i = 0
parameters = {'zf': dict(), 'dau': dict()}
parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[i,:3,:3]), Tracker_ZF_midhand.T_opt_ct[i,:3,3]])
parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[i,:3,:3]), Tracker_DAU_DIP.T_opt_ct[i,:3,3]])
parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[i,:3,:3]), Tracker_DAU_MCP.T_opt_ct[i,:3,3]])

with open("./mujoco/generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template_simple.xml", "./mujoco/generated_parameters.yaml")
print("Model updated!")