# %%
from copy import copy
import csv
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from stl import mesh
from pyquaternion import Quaternion
import xml.etree.ElementTree as ET
import hydra
from hydra.core.config_store import ConfigStore
from config import HandSettings, TrackSettings
from dm_control import mujoco
from tqdm import tqdm

cs = ConfigStore.instance()
cs.store(name='hand_config', node=HandSettings)

def sort_points_relative(list_points_1, list_points_2):
    # Convert lists to numpy arrays
    list_points_1 = np.array(list_points_1)
    list_points_2 = np.array(list_points_2)
    # Compute pairwise distances for each list
    distances_1 = squareform(pdist(list_points_1))
    distances_2 = squareform(pdist(list_points_2))
    # Sum the distances for each point
    sum_distances_1 = np.sum(distances_1, axis=1)
    sum_distances_2 = np.sum(distances_2, axis=1)
    # Get the sorted indices based on the sum of distances
    sorted_indices_1 = np.argsort(sum_distances_1)
    sorted_indices_2 = np.argsort(sum_distances_2)
    # Sort the points based on the computed indices
    sorted_list_points_1 = list_points_1[sorted_indices_1]
    sorted_list_points_2 = list_points_2[sorted_indices_2]
    return sorted_list_points_1.tolist(), sorted_list_points_2.tolist()

def kabsch(P, Q):
    """Calculate the optimal rigid transformation matrix from Q to P using Kabsch algorithm"""
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = np.dot(P_centered.T, Q_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_Q - np.dot(centroid_P, R.T)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inverse_transformation_matrix(T):
    # Extract the rotation and translation parts
    R = T[:3, :3]
    t = T[:3, 3]
    # Compute the inverse rotation as the transpose of R
    R_inv = R.T
    # Compute the inverse translation as the negative dot product of R_inv and t
    t_inv = -np.dot(R_inv, t)
    # Create the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def csv_test_load(df, tracker_name):
    """get the optitrack measure data for the tracker"""
    start_coloum = df.columns.get_loc(tracker_name)
    data = df.values[3:, start_coloum:start_coloum+7]
    data = np.array([list(map(float, i)) for i in data])
    return data

def update_all_pos_quat(root, name, pos, quat):
    """Updates the position and quaternion of an element with the specified name."""
    for elem in root[3]:
        if elem.attrib['name'] == name:
            elem.attrib['pos'] = write_arr_xml(pos)
            elem.attrib['quat'] = write_arr_xml(quat)

def write_arr_xml(arr):
    """Converts an array of elements to a space-separated string."""
    return ' '.join(str(elem) for elem in arr)

class Tracker:
    def _init_(self, cfg: TrackSettings, df: pd.DataFrame, verbose=False):
        """init a tracker object"""
        self.cfg = cfg
        self.def_path = cfg.def_path
        self.ct_path = cfg.ct_path
        self.your_mesh = mesh.Mesh.from_file(cfg.stl_path)
        self.measure_data = csv_test_load(df, cfg.csv_tracker_name)
        if verbose:
            plt.figure(cfg.def_path)
            plt.subplot(211)
            plt.plot(self.measure_data[:, :4])
            plt.subplot(212)
            plt.plot(self.measure_data[:, 4:])
            plt.show()
        # read the point lists
        self.marker_pos_def = self.read_markerdata(self.def_path)
        self.read_ctdata()
        # sort the point lists by relative distance
        self.marker_pos_def, self.marker_pos_ct = sort_points_relative(
            self.marker_pos_def, self.marker_pos_ct)
        # now calculate transformation matrix
        self.t_def_ct = kabsch(
            self.marker_pos_ct, self.marker_pos_def)
        # and its inverse
        self.t_ct_def = inverse_transformation_matrix(self.t_def_ct)

    def read_markerdata(self, path):
        """read the data from the csv file"""
        coordinates = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the first row (header)
            next(reader)  # Skip the second row (version)
            next(reader)  # Skip the third row (name)
            next(reader)  # Skip the fourth row (ID)
            next(reader)  # Skip the fifth row (color)
            next(reader)  # Skip the sixth row (units)
            for row in reader:
                coordinates.append(
                    [float(row[2]), float(row[3]), float(row[4])])
        return coordinates
    
    def update_tmat(self, t_index):
        rel_line = self.measure_data[t_index, :]
        qx, qy, qz, qw, x, y, z = rel_line
        quat = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)
        if quat.w < 0:
            quat *= -1
        v = [x, y, z]
        t_mat = np.eye(4)
        t_mat[:3, :3] = quat.rotation_matrix
        t_mat[:3, 3] = v
        self.t_opt_def = t_mat
        self.t_def_opt = inverse_transformation_matrix(t_mat)
        self.tmat_t = self.t_opt_def @ self.t_def_ct
        return self.get_quaternion_pos()
    
    def read_ctdata(self):
        """read the data of the marker points from the ct scan"""
        # 1. load mounting points
        with open(self.ct_path) as jsonfile:
            data = json.load(jsonfile)
        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        self.marker_pos_ct = [point['position'] for point in point_data]

    def get_quaternion_pos(self, offz=0.7):
        quat = Quaternion._from_matrix(self.tmat_t[:3, :3])
        quat = quat.elements
        vec = copy(self.tmat_t[:3, 3])
        vec *= 0.001
        vec[2] += offz
        return quat, vec
    
class Finger:
    def _init_(self, cfg: HandSettings):
        self.cfg = cfg
        self.tree = ET.parse(cfg.mujoco_file)
        self.root = self.tree.getroot()
        self.df = pd.read_csv(cfg.optitrack_file, header=2, low_memory=False)
        self.tracker_dau_dip = Tracker(cfg.dau_dip, self.df)
        self.tracker_dau_mcp = Tracker(cfg.dau_mcp, self.df)
        self.tracker_zf_mcp = Tracker(cfg.zf_dip, self.df)
        self.tracker_zf_dip = Tracker(cfg.zf_mcp, self.df)
    def update(self, t_index):
        quat1, vec1 = self.tracker_dau_dip.update_tmat(t_index)
        update_all_pos_quat(self.root, 'DAU_DIP', vec1, quat1)
        quat2, vec2 = self.tracker_dau_mcp.update_tmat(t_index)
        update_all_pos_quat(self.root, 'DAU_MCP', vec2, quat2)
        quat3, vec3 = self.tracker_zf_mcp.update_tmat(t_index)
        update_all_pos_quat(self.root, 'ZF_MCP', vec3, quat3)
        quat4, vec4 = self.tracker_zf_dip.update_tmat(t_index)
        update_all_pos_quat(self.root, 'ZF_DIP', vec4, quat4)
        self.tree.write(self.cfg.mujoco_file)
    def load(self):
        self.model = recursive_loading(
            './scene.xml', template_mode=False, path_ext='./')
        self.physics = mujoco.Physics.from_xml_string(self.model)
    def get_pixels(self):
        pixels = self.physics.render(height=1024, width=1024)
        return pixels
    def make_video(self, fps, start_pos, end_pos):
        frames = []
        for index in tqdm(range(start_pos, end_pos)):
            self.update(index)
            self.load()
            frames.append(self.get_pixels())
        display_video(frames, framerate=fps)

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
    anim.save('./test.gif')

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
# %%
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: HandSettings):
    dau = Finger(cfg)
    dau.make_video(60, 700, 720)
    dau.update(0)
if _name_ == '_main_':
    main()
# %%