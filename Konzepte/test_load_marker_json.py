# %%
import numpy as np
import sys
sys.path.append('./')
sys.path.append('./mujoco')
from . import transformation_functions as tf
import my_write_parameters as mwp
import my_model as mwj
import yaml


path='Data/optitrack-20230130-235000.json'

df = tf.load_marker_from_json(path=path)
for i in range(0, len(df.columns)):
    if i % 3 == 0:
        if df.columns[i] == 51533 or df.columns[i] == 54597 or df.columns[i] == 54615:
            print('Marker ID: ', df.columns[i])
        print(df.columns[i])


print('-'*30)
t = 0

parameters = {'marker': dict()}
j = 0
for i in range(0, len(df.columns)):
    if i%3 == 0:
        parameters['marker'][str(j)] = mwp.build_parameters([[1,0,0,0], df.iloc[t,i:i+3]])
        j += 1
        print(df.iloc[t,i:i+3])


with open("./mujoco/json_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

model = mwj.MujocoFingerModel("./mujoco/json_template.xml", "./mujoco/json_parameters.yaml")
print("Model updated!")

# %%
