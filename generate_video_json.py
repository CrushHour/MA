import numpy as np
import sys
sys.path.append('./')
sys.path.append('./mujoco')
sys.path.append(r'C:/GitHub/MA/mujoco')
sys.path.append(r'C:/GitHub/MA')
import transformation_functions as tf
import my_write_parameters as mwp
import my_model as mwj
import yaml
import mediapy as media
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyquaternion import Quaternion
from tqdm import tqdm
from dm_control import mujoco as dm_mujoco



class MujocoFingerModel:

    def __init__(self, path) -> None:
        self.Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
        self.Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
        self.Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
        self.Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)
        self.Marker_DAU = tf.marker_bone(finger_name='DAU_PIP',test_path=test_metadata['path'])
        self.Marker_ZF_intermedial = tf.marker_bone(finger_name="ZF_PIP",test_path=test_metadata['path'])
        self.ZF_MCP = tf.marker_bone(finger_name="ZF_MCP",test_path=test_metadata['path'], init_marker_ID='')

        self.construct_all_T_opt_ct()

        self.end_pos = self.Tracker_ZF_DIP.T_opt_ct.shape[0]
        self.offset = 0.85*np.array(self.Tracker_ZF_midhand.T_opt_ct[0,:3,3])

        self.title = path.split('/')[-1].split('.')[0]

        try:
            self.df, time = tf.phoenix_test_load(path=path)
        except Exception as e:
            print(e)
            print("Error loading json file!")
            return
    

    
    def construct_all_T_opt_ct(self):
        for t in tqdm(range(len(self.Marker_DAU.opt_marker_trace))):
            self.Marker_DAU.T_opt_ct[t] = self.construct_marker_rot([self.Tracker_DAU_DIP.T_proxi_innen_opt[t,:3,3], self.Tracker_DAU_DIP.T_proxi_aussen_opt[t,:3,3],self.Tracker_DAU_MCP.T_dist_innen_opt[t,:3,3],self.Tracker_DAU_MCP.T_dist_aussen_opt[t,:3,3]],\
                                                        [self.Tracker_DAU_DIP.T_proxi_innen_CT[:3,3], self.Tracker_DAU_DIP.T_proxi_aussen_CT[:3,3],self.Tracker_DAU_MCP.T_dist_innen_CT[:3,3],self.Tracker_DAU_MCP.T_dist_aussen_CT[:3,3]])
            self.Marker_DAU.update_joints(t)

            self.Marker_ZF_intermedial.T_opt_ct[t] = self.Tracker_ZF_DIP.T_opt_ct[t]

            # update loop auf basis aller bekannten Punkte
            for j in range(3):
                self.Marker_ZF_intermedial.update_joints(t)

                self.ZF_MCP.T_opt_ct[t] = self.construct_marker_rot([self.Tracker_ZF_midhand.T_dist_innen_opt[t,:3,3],self.Tracker_ZF_midhand.T_dist_aussen_opt[t,:3,3],self.Marker_ZF_intermedial.T_proxi_opt[t,0,:3,3],self.Marker_ZF_intermedial.T_proxi_opt[t,1,:3,3]], \
                                                        [self.Tracker_ZF_midhand.T_dist_innen_CT[:3,3],self.Tracker_ZF_midhand.T_dist_aussen_CT[:3,3],self.Marker_ZF_intermedial.T_proxi_CT[0,:3,3], self.Marker_ZF_intermedial.T_proxi_CT[1,:3,3]])
                self.ZF_MCP.update_joints(t)

                self.Marker_ZF_intermedial.T_opt_ct[t] = self.construct_marker_rot([self.Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],self.Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3], self.ZF_MCP.T_dist_opt[t,0,:3,3], self.ZF_MCP.T_dist_opt[t,1,:3,3]], \
                                                                [self.Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], self.Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3], self.ZF_MCP.T_dist_CT[0,:3,3], self.ZF_MCP.T_dist_CT[1,:3,3]])

    
    def update(self, i, offset = np.array([0,0,0])):
        parameters = {'marker': dict(), 'zf': dict(), 'dau': dict()}

        # STL
        parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_DIP.T_opt_ct[i,:3,:3]), self.Tracker_ZF_DIP.T_opt_ct[i,:3,3]])
        parameters['zf']['pip'] = mwp.build_parameters([Quaternion(matrix=self.Marker_ZF_intermedial.T_opt_ct[i,:3,:3]), self.Marker_ZF_intermedial.T_opt_ct[i,:3,3]])
        parameters['zf']['mcp'] = mwp.build_parameters([Quaternion(matrix=self.ZF_MCP.T_opt_ct[i,:3,:3]) ,self.ZF_MCP.T_opt_ct[i,:3,3]])
        parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_midhand.T_opt_ct[i,:3,:3]), self.Tracker_ZF_midhand.T_opt_ct[i,:3,3]])

        # green, red, yellow, white, blue balls
        parameters['zf']['dip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_ZF_DIP.T_proxi_aussen_opt[i,:3,3]]) 
        parameters['zf']['dip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_ZF_DIP.T_proxi_innen_opt[i,:3,3]]) 
        parameters['zf']['pip_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], self.Marker_ZF_intermedial.T_proxi_opt[i,1,:3,3]]) 
        parameters['zf']['pip_joint_innen' ] = mwp.build_parameters([[1,0,0,0], self.Marker_ZF_intermedial.T_proxi_opt[i,0,:3,3]]) 
        parameters['zf']['mcp_joint_aussen' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_ZF_midhand.T_dist_aussen_opt[i,:3,3]]) 
        parameters['zf']['mcp_joint_innen' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_ZF_midhand.T_dist_innen_opt[i,:3,3]]) 
        parameters['zf']['pip_marker'] = mwp.build_parameters([[1,0,0,0], self.Marker_ZF_intermedial.opt_marker_trace[i]]) 

        # STL
        parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_DIP.T_opt_ct[i,:3,:3]), self.Tracker_DAU_DIP.T_opt_ct[i,:3,3]])
        parameters['dau']['pip'] = mwp.build_parameters([Quaternion(matrix=self.Marker_DAU.T_opt_ct[i,:3,:3]), self.Marker_DAU.T_opt_ct[i,:3,3]])
        parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_MCP.T_opt_ct[i,:3,:3]), self.Tracker_DAU_MCP.T_opt_ct[i,:3,3]])

        # green, yellow balls
        parameters['dau']['pip_joint' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_DAU_DIP.T_proxi_opt[i,:3,3]]) 
        parameters['dau']['mcp_joint' ] = mwp.build_parameters([[1,0,0,0], self.Tracker_DAU_MCP.T_dist_opt[i,:3,3]]) 
        parameters['dau']['pip_marker'] = mwp.build_parameters([[1,0,0,0], self.Marker_DAU.opt_marker_trace[i]])

        
        with open("./mujoco/generated_parameters.yaml", "w") as outfile:
            yaml.dump(parameters, outfile)
        
        self.model = mwj.MujocoFingerModel("./mujoco/my_tendom_finger_template.xml", "./mujoco/generated_parameters.yaml")

        self.physics_model = self.recursive_loading(xml_path = './finger_control.xml',path_ext='./', template_mode=False)

        self.physics = dm_mujoco.Physics.from_xml_string(self.physics_model)

    def get_pixels(self):
        pixels = self.physics.render(height=1024, width=1024, camera_id=0)
        return pixels
    
    def make_video(self, fps, start_pos=0):
        stepsize = int(120/fps)
        frames = []
        for index in tqdm(range(start_pos, self.end_pos, stepsize), desc='Rendering frames'):
            self.update(index, self.offset)
            frames.append(self.get_pixels())
        display_video(frames, framerate=fps, title=self.title)

    def recursive_loading(self, xml_path, path_ext='../', st_s='<include file=', end_s='/>', template_mode=False):
        """recursively load subfiles"""
        with open(xml_path, "r") as stream:
            xml_string = stream.read()
        xml_string = xml_string.replace("./", path_ext)
        extra_file, start_p, end_p = find_between(xml_string, st_s, end_s)
        if template_mode:
            filename = extra_file.split('/')[-1].split('.')[0]
            extra_file = f'{path_ext}{filename}_template.xml'
        if len(extra_file) > 0:
            extra_string = recursive_loading(
                extra_file, path_ext=path_ext)
            spos = extra_string.index('<mujoco model=')
            end = extra_string.index('>', spos)
            extra_string = extra_string[:spos] + extra_string[end:]
            extra_string = extra_string.replace('</mujoco>', '')
            xml_string = xml_string[:start_p] + extra_string + xml_string[end_p:]
        return xml_string

    def construct_marker_rot(self, opt_info, ct_info):
        '''Do kabsch with points from different sources. Points must be in the same order.'''
        T = self.Tracker_DAU_DIP.kabsch(ct_info, opt_info)
        return T

def find_between(s: str, first: str, last: str):
    """helper for string preformatting"""
    try:
        start = s.index(first) + len(first)
        start_pos = s.index(first)
        end = s.index(last, start)
        return s[start:end].replace('"', ''), start_pos, end
    except ValueError:
        return "", "", ""

def recursive_loading(xml_path, path_ext='../', st_s='<include file=', end_s='/>', template_mode=False):
    """recursively load subfiles"""
    with open(xml_path, "r") as stream:
        xml_string = stream.read()
    xml_string = xml_string.replace("./", path_ext)
    extra_file, start_p, end_p = find_between(xml_string, st_s, end_s)
    if template_mode:
        filename = extra_file.split('/')[-1].split('.')[0]
        extra_file = f'{path_ext}{filename}_template.xml'
    if len(extra_file) > 0:
        extra_string = recursive_loading(
            extra_file, path_ext=path_ext)
        spos = extra_string.index('<mujoco model=')
        end = extra_string.index('>', spos)
        extra_string = extra_string[:spos] + extra_string[end:]
        extra_string = extra_string.replace('</mujoco>', '')
        xml_string = xml_string[:start_p] + extra_string + xml_string[end_p:]
    return xml_string

def display_video(frames, framerate=60, dpi=600, title='Video'):
    height, width, _ = frames[0].shape
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    anim.save('./plots/' + title + '.mp4')
    plt.close(fig)

if __name__ == '__main__':
    test_metadata = tf.get_test_metadata('2023_01_31_18_12_48.json')
    hand_metadata = tf.get_json('hand_metadata.json')

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    finger = MujocoFingerModel(test_metadata['path'])
    finger.make_video(60, 0)