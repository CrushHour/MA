import sys
import os
sys.path.append('./mujoco')
import my_write_parameters as mwp
from pyquaternion import Quaternion
import my_model as mwj
import yaml
import numpy as np
import json

# a function that loads a folder of json files and returns a dictionary of dictionaries with the filenames as keys
def load_json_folder(folder_path):
    file_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith("AXIS.mrk.json"):
            with open(os.path.join(folder_path, filename), "r") as read_file:
                current_dict = json.load(read_file)
                file_dict[filename[:-14]] = current_dict['markups'][0]['controlPoints']
    return file_dict

d = load_json_folder("Data\\Slicer3D\\Joints")

if __name__ == "__main__":
    parameters = {'zf': dict(), 'dau': dict()}

    # STL
    parameters['zf']['dip'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])
    parameters['zf']['pip'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])
    parameters['zf']['mcp'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])
    parameters['zf']['midhand'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])

    # green, red, yellow, white, blue balls
    parameters['zf']['dip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_DIP'][0]['position']]) 
    parameters['zf']['dip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_DIP'][1]['position']]) 
    parameters['zf']['pip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_PIP'][0]['position']]) 
    parameters['zf']['pip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_PIP'][1]['position']]) 
    parameters['zf']['mcp_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_MCP'][0]['position']]) 
    parameters['zf']['mcp_joint_innen' ] = mwp.build_parameters([[1,0,0,0], d['ZF_MCP'][1]['position']]) 

    # STL
    parameters['dau']['dip'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])
    parameters['dau']['pip'] = mwp.build_parameters([[1,0,0,0], [0,0,0]])
    parameters['dau']['mcp' ] = mwp.build_parameters([[1,0,0,0], [0,0,0]])

    parameters['dau']['dip_joint_aussen'] = mwp.build_parameters([[1,0,0,0], d['DAU_DIP'][0]['position']]) 
    parameters['dau']['dip_joint_innen'] = mwp.build_parameters([[1,0,0,0], d['DAU_DIP'][1]['position']]) 
    parameters['dau']['mcp_joint_aussen'] = mwp.build_parameters([[1,0,0,0], d['DAU_PIP'][0]['position']]) 
    parameters['dau']['mcp_joint_innen'] = mwp.build_parameters([[1,0,0,0], d['DAU_PIP'][1]['position']])

    with open("./mujoco/generated_parameters.yaml", "w") as outfile:
        yaml.dump(parameters, outfile)

    model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template_white.xml", "./mujoco/generated_parameters.yaml")
    print("Model updated!")