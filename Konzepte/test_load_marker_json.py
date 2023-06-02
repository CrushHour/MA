# %%
import numpy as np
import sys
sys.path.append('./')
sys.path.append('./mujoco')
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

path='Data/optitrack-20230130-235000.json'

Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)

df = tf.json_test_load(path=path)
#df.drop('Index', axis=1, inplace=True)
for i in range(0, len(df.columns)):
    if i % 7 == 4:
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
class MujocoFingerModel:

    def __init__(self, path) -> None:
        self.Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',path)
        self.Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',path)
        self.Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',path)
        self.Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',path)

        self.end_pos = self.Tracker_ZF_DIP.T_opt_ct.shape[0]
        self.end_pos = 100
    
    def update(self, t):
        parameters = {'marker': dict(), 'zf': dict(), 'dau': dict()}

        parameters['zf']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_DIP.T_opt_ct[t,:3,:3]), self.Tracker_ZF_DIP.T_opt_ct[t,:3,3]])
        parameters['zf']['midhand'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_ZF_midhand.T_opt_ct[t,:3,:3]), self.Tracker_ZF_midhand.T_opt_ct[t,:3,3]])
        parameters['dau']['dip'] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_DIP.T_opt_ct[t,:3,:3]), self.Tracker_DAU_DIP.T_opt_ct[t,:3,3]])
        parameters['dau']['mcp' ] = mwp.build_parameters([Quaternion(matrix=self.Tracker_DAU_MCP.T_opt_ct[t,:3,:3]), self.Tracker_DAU_MCP.T_opt_ct[t,:3,3]])

        j = 0
        for i in range(0, len(df.columns)):
            if i%7 == 0:
                parameters['marker'][str(j)] = mwp.build_parameters([[1,0,0,0], df.iloc[t,i+4:i+7]])
                j += 1
        
        with open("./mujoco/json_parameters.yaml", "w") as outfile:
            yaml.dump(parameters, outfile)
        
        self.model = mwj.MujocoFingerModel("./mujoco/json_template.xml", "./mujoco/json_parameters.yaml")

        self.physics_model = self.recursive_loading(xml_path = './finger_control.xml',path_ext='./', template_mode=False)

        self.physics = dm_mujoco.Physics.from_xml_string(self.physics_model)
        model = mujoco.MjModel.from_xml_string(self.physics_model)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)

        media.show_image(renderer.render())

    def get_pixels(self):
        #pixels = self.physics.render(height=1024, width=1024)
        pixels = self.physics.render()
        return pixels
    
    def make_video(self, fps, start_pos=0):
        frames = []
        for index in tqdm(range(start_pos, self.end_pos)):
            self.update(index)
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

def load():
    model = recursive_loading(
        './json.xml', template_mode=False, path_ext='./')
    physics = mujoco.Physics.from_xml_string(model)
    return model, physics

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
    anim.save('./plots/test.gif')

def compute_camera_matrix(renderer, data):
  """Returns the 3x4 camera matrix."""
  # If the camera is a 'free' camera, we get its position and orientation
  # from the scene data structure. It is a stereo camera, so we average over
  # the left and right channels. Note: we call `self.update()` in order to
  # ensure that the contents of `scene.camera` are correct.
  renderer.update_scene(data)
  pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
  z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
  y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
  rot = np.vstack((np.cross(y, z), y, z))
  fov = model.vis.global_.fovy

  # Translation matrix (4x4).
  translation = np.eye(4)
  translation[0:3, 3] = -pos

  # Rotation matrix (4x4).
  rotation = np.eye(4)
  rotation[0:3, 0:3] = rot

  # Focal transformation matrix (3x4).
  focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
  focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

  # Image matrix (3x3).
  image = np.eye(3)
  image[0, 2] = (renderer.width - 1) / 2.0
  image[1, 2] = (renderer.height - 1) / 2.0
  return image @ focal @ rotation @ translation

finger = MujocoFingerModel(path)
finger.make_video(60, 0)