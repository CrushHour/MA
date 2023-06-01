# %%
import numpy as np
import sys
sys.path.append('./')
sys.path.append('./mujoco')
import transformation_functions as tf
import my_write_parameters as mwp
import my_model as mwj
import yaml
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


path='Data/optitrack-20230130-235000.json'

Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)

df = tf.json_test_load(path=path)
#df.drop('Index', axis=1, inplace=True)
for i in range(0, len(df.columns)):
    if i % 7 == 4:
        if df.columns[i] == 51533 or df.columns[i] == 54597 or df.columns[i] == 54615:
            print('Marker ID: ', df.columns[i])
        print(df.columns[i])

# %%
print('-'*30)
t = 0
show_plots = False

parameters = {'marker': dict(), 'zf': dict(), 'dau': dict()}

parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_DIP.T_opt_ct[t,:3,:3]), Tracker_ZF_DIP.T_opt_ct[t,:3,3]])
parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[t,:3,:3]), Tracker_ZF_midhand.T_opt_ct[t,:3,3]])
parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[t,:3,:3]), Tracker_DAU_DIP.T_opt_ct[t,:3,3]])
parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[t,:3,:3]), Tracker_DAU_MCP.T_opt_ct[t,:3,3]])

j = 0
for i in range(0, len(df.columns)):
    if i%7 == 0:
        parameters['marker'][str(j)] = mwp.build_parameters([[1,0,0,0], df.iloc[t,i+4:i+7]])
        j += 1
        if show_plots:
            plt.plot(df.iloc[:,i+4:i+7])
            plt.legend(['x','y','z'])
            plt.title('Marker ID: ' + str(df.columns[i]))
            plt.show()
            #plt.savefig('./plots/optitrack-20230130-235000/' + str(df.columns[i]) + '.png')
            plt.close()

with open("./mujoco/json_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/json_template.xml", "./mujoco/json_parameters.yaml")
print("Model updated!")

# %%
