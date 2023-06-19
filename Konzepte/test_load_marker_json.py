# %%
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
import mujoco

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

path='Data/optitrack-20230130-234800.json'
path='Data/test_01_31/2023_01_31_18_12_48.json'

indipendent = False

if indipendent:
    Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
    Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
    Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
    Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)

    offset = 0.8*np.array(Tracker_ZF_midhand.T_opt_ct[0,:3,3])
    

    df = tf.json_test_load(path=path)
    #df.drop('Index', axis=1, inplace=True)
    for i in range(0, len(df.columns)):
        if i % 7 == 4:
            print(df.columns[i])


    print('-'*30)
    t = 0
    show_plots = False

    print(Tracker_ZF_midhand.T_opt_ct[t,:3,3]-offset)

    parameters = {'marker': dict(), 'zf': dict(), 'dau': dict()}

    parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_DIP.T_opt_ct[t,:3,:3]), Tracker_ZF_DIP.T_opt_ct[t,:3,3]-offset])
    parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[t,:3,:3]), Tracker_ZF_midhand.T_opt_ct[t,:3,3]-offset])
    parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[t,:3,:3]), Tracker_DAU_DIP.T_opt_ct[t,:3,3]-offset])
    parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[t,:3,:3]), Tracker_DAU_MCP.T_opt_ct[t,:3,3]-offset])

    j = 0
    
    for i in range(0, len(df.columns)):
        if i%7 == 0:
            parameters['marker'][str(j)] = mwp.build_parameters([[1,0,0,0], df.iloc[t,i+4:i+7]-offset])
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
class MujocoFingerModel:

    def __init__(self, path) -> None:
        self.Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
        self.Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
        self.Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
        self.Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)

        self.end_pos = self.Tracker_ZF_DIP.T_opt_ct.shape[0]
        self.offset = 0.85*np.array(self.Tracker_ZF_midhand.T_opt_ct[0,:3,3])

        try:
            self.df = tf.json_test_load(path=path)
        except Exception as e:
            print(e)
            print("Error loading json file!")
            return
    
    def update(self, t, offset = np.array([0,0,0])):
        parameters = {'marker': dict(), 'zf': dict(), 'dau': dict()}

        parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_DIP.T_opt_ct[t,:3,:3]), self.Tracker_ZF_DIP.T_opt_ct[t,:3,3]-offset])
        parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_midhand.T_opt_ct[t,:3,:3]), self.Tracker_ZF_midhand.T_opt_ct[t,:3,3]-offset])
        parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_DIP.T_opt_ct[t,:3,:3]), self.Tracker_DAU_DIP.T_opt_ct[t,:3,3]-offset])
        parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_MCP.T_opt_ct[t,:3,:3]), self.Tracker_DAU_MCP.T_opt_ct[t,:3,3]-offset])

        j = 0
        for i in range(0, len(self.df.columns)):
            if i%7 == 0:
                parameters['marker'][str(j)] = mwp.build_parameters([[1,0,0,0], self.df.iloc[t,i+4:i+7]-offset])
                j += 1
        
        with open("./mujoco/json_parameters.yaml", "w") as outfile:
            yaml.dump(parameters, outfile)
        
        self.model = mwj.MujocoFingerModel("./mujoco/json_template.xml", "./mujoco/json_parameters.yaml")

        self.physics_model = self.recursive_loading(xml_path = './finger_control.xml',path_ext='./', template_mode=False)

        self.physics = dm_mujoco.Physics.from_xml_string(self.physics_model)
        #model = mujoco.MjModel.from_xml_string(self.physics_model)
        #data = mujoco.MjData(model)
        #renderer = mujoco.Renderer(model)
        #mujoco.mj_forward(model, data)
        #renderer.update_scene(data)

        #media.show_image(renderer.render())

    def get_pixels(self):
        pixels = self.physics.render(height=480, width=640)
        return pixels
    
    def make_video(self, fps, start_pos=0):
        stepsize = int(120/fps)
        frames = []
        for index in tqdm(range(start_pos, self.end_pos, stepsize), desc='Rendering frames'):
            self.update(index, self.offset)
            frames.append(self.get_pixels())
        display_video(frames, framerate=fps)

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

# %%
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

def display_video(frames, framerate=60, dpi=600):
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
    anim.save('./plots/test_2348.mp4')

if __name__ == '__main__':
    finger = MujocoFingerModel(path)
    finger.make_video(60, 0)