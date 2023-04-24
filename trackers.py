# %% File for handling the conversion between optitrack stream and Tracker positions
import math
import csv
import json
import numpy as np
import torch
from pyquaternion.quaternion import Quaternion
from datasplit import ObservationHandler


def return_sorted_points(points1, points2):
    """summarize the functions"""
    pairs1 = sort_points_by_dis(points1)
    pairs2 = sort_points_by_dis(points2)
    return compare_point_lists(pairs1, points1, pairs2, points2)


def sort_points_by_dis(points):
    n = len(points)
    pairs = []
    for i in range(n):
        for j in range(n - 1, -1 + i, -1):
            if j != i:
                #dx = points[j][0] - points[i][0]
                #dy = points[j][1] - points[i][1]
                #dz = points[j][2] - points[i][2]
                #d_diff = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                d_diff = np.linalg.norm(np.array(points[j])-np.array(points[i]))
                # erstelle Liste die alle Punkte distanzen, plus euklidische Distanz enthält
                pairs.append([points[j], points[i], d_diff])
    # Sortiere Punktepaare nach euklidischer Distanz. Aufsteigend von kurz nach lang.
    pairs.sort(key=lambda x: x[2])

    return pairs


def compare_point_lists(pairs1, points1, pairs2, points2):
    '''Lese Punktepaare, und Punktewolken ein. Vergleiche Positionen der Punkte anhand 
    der Stellen in den Punktepaaren, in denen sie
    vorkommen.'''
    distance_value_in_points1 = [[], [], [], [], []]
    distance_value_in_points2 = [[], [], [], [], []]

    '''Erstelle für jeden Punkt eine Liste, in der Steht in welcher Distanz er vokommt.'''
    for i in range(len(pairs1)):
        for j in range(len(points1)):

            print('points1:')
            print(points1[j])
            print('pairs1:')
            print(pairs1[i])


            # Ich habe hier fast eine Woche verschwendet. 
            # Tut mir leid, aber ich konnte den Value Error nicht beheben, obwohl die scheiß if clause nur ein einfaches True oder False zurück gegben hat.
            # Ich hasse alles.
            #if (points1[j] in pairs1[i]).bit_length() > 0:
            try: 
                pairs1[i].index(points1[j])
                distance_value_in_points1[j].append(i + 1)
            except:
                pass

            # if (points2[j] in pairs2[i]).bit_length() > 0:
            try: 
                pairs2[i].index(points2[j])
                distance_value_in_points2[j].append(i + 1)
            except:
                pass

    '''An dieser Stelle soll die Liste der Punkte aus dem CT (2) anhand der Punkte aus dem Opti-Export (1)
    sortiert werden. Dafür wird der Index einer Distanz Index Kombi von (2) in (1) gesucht und der Index gepseichert.
    Anhand der entstehenden Liste von Indexen werden die Punkte von (2) umsortiert.'''''
    index_list = []
    for i in range(len(points1)):
        index_list.append(distance_value_in_points1.index(
            distance_value_in_points2[i]))

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    points_2_out = [x for _, x in sorted(zip(index_list, points2))]

    return points1, points_2_out


class TMatrix(object):
    """The Ligntning Module containing the logic of 4x4 Transformation matrices"""

    def __init__(self, quat: Quaternion, pos: np.ndarray):
        """define all initial parameters"""
        super(TMatrix, self).__init__()
        # define matrix and rotation
        self.quat = torch.tensor(quat.q).float()
        self.rotmat = torch.tensor(quat.rotation_matrix).float()
        self.pos = torch.tensor(pos).float()

        self.forw = self.get_matrix()
        self.backw = self.get_matrix(inverse=True)

    def norm(self):
        """ensure quat is always normalised first"""
        sum_sq = torch.dot(self.quat, self.quat)
        norm = torch.sqrt(sum_sq)
        q_norm = self.quat / norm
        return q_norm

    def quat_to_rot(self):
        """add quat to rot to computation graph"""
        quat_norm = self.norm()
        qw, qx, qy, qz = quat_norm

        matrix = torch.zeros(3, 3)

        matrix[0, 0] = 1. - 2. * qy ** 2 - 2. * qz ** 2
        matrix[1, 1] = 1. - 2. * qx ** 2 - 2. * qz ** 2
        matrix[2, 2] = 1. - 2. * qx ** 2 - 2. * qy ** 2

        matrix[0, 1] = 2. * qx * qy - 2. * qz * qw
        matrix[1, 0] = 2. * qx * qy + 2. * qz * qw

        matrix[0, 2] = 2. * qx * qz + 2 * qy * qw
        matrix[2, 0] = 2. * qx * qz - 2 * qy * qw

        matrix[1, 2] = 2. * qy * qz - 2. * qx * qw
        matrix[2, 1] = 2. * qy * qz + 2. * qx * qw

        return matrix

    def get_pos(self):
        """assign pos"""
        cur_pos = torch.zeros(3)
        cur_pos[0] = self.pos[0]
        cur_pos[1] = self.pos[1]
        cur_pos[2] = self.pos[2]
        return cur_pos

    def get_matrix(self, inverse=False):
        """ge the complete transformation matrix"""
        matrix = torch.zeros(4, 4)
        rot_mat = self.quat_to_rot().T if inverse else self.quat_to_rot()
        matrix[:3, :3] = rot_mat
        matrix[3, 3] = 1
        matrix[:3, 3] = - \
            torch.matmul(rot_mat, self.get_pos()
                         ) if inverse else self.get_pos()
        return matrix

    def forward(self, x, inverse=False):
        """transform vector with tmat, depending on inverse or not"""
        # easier to calculate
        matrix = self.backw if inverse else self.forw
        return torch.matmul(matrix, x)

    def info(self):
        """print the info of 4x4 transformations, pos and quat"""
        print(self.quat.detach().numpy())
        print(self.pos.detach().numpy())
        print(self.quat_to_rot().detach().numpy())
        print(self.get_matrix().detach().numpy())


# %%


class Tracker(object):
    """class to handle the"""

    def __init__(self, id_num: int, defname: str, ctname=None, finger_name=None):
        """init the tracker
        1. read files from pointlist
        """
        self.metadata = self.get_metadata()
        self.id_num = id_num
        self.defname = defname
        self.ctname = ctname
        self.finger_name = finger_name
        
        # read definition file into memory
        self.read_markerdata()

        
        if ctname is not None: 
            # read jsonfile from ct into memory
            self.read_ctdata()
            # catch the case where the ct file is about a marker instead of a tracker
            if self.metadata["tracking"] == "tracker":
                self.calculate_transformation_matrix()
                # this assumes that the tracker definition file has coordinates in the same
                # system as the recording file? - Julian
                self.t_ct_tr = self.t_ct_def
        else:
            # define a test scenario
            self.perform_test()

        
    
    def get_metadata(self):
        '''Returns the metadata of the Phalanx.'''
        with open('hand_metadata.json') as json_data:
            d = json.load(json_data)
            metadata = d[self.finger_name]
            json_data.close()
        return metadata
    
    def perform_test(self):
        """function to perform the test
        # take the 
        """
        q_init_f = Quaternion(np.random.randn(4))
        p_init_f = np.array(np.random.randn(3))
        self.transform_fake = TMatrix(q_init_f, p_init_f)
        print(self.transform_fake.get_matrix())

        # now transform the markers by the fake matrix:
        self.marker_pos_ct = []
        for point in self.marker_pos_def:
            torch_vec = torch.tensor([point[0], point[1], point[2], 1])
            trans_vec = self.transform_fake.forward(torch_vec)
            trans_vec = trans_vec.cpu().detach().numpy().tolist()
            self.marker_pos_ct.append(
                [trans_vec[0], trans_vec[1], trans_vec[2]])
        points = np.random.permutation(np.array(self.marker_pos_ct)).tolist()
        self.marker_pos_ct = points
        _, self.marker_pos_ct = return_sorted_points(
            self.marker_pos_def, self.marker_pos_ct)
        if len(self.marker_pos_ct) == len(self.marker_pos_def):
            _ = self.calculate_transformation_matrix()
            print(np.round(self.t_ct_def, decimals=4))
            print(np.round(self.t_def_ct, decimals=4))

    def read_ctdata(self):
        """read the data of the marker points from the ct scan"""
        # 1. load mounting points
        with open(self.ctname) as jsonfile:
            data = json.load(jsonfile)
        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        self.marker_pos_ct = [point['position'] for point in point_data]

    def read_markerdata(self):
        """read the data from the csv file"""
        coordinates = []
        with open(self.defname, 'r') as file:
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
        self.marker_pos_def = coordinates
        # reader schließen?
        file.close()

    # von def in CT
    def calculate_transformation_matrix(self):
        """the required tranformation matrix between system 1 and 2"""
        markers2 = self.marker_pos_def
        markers1 = self.marker_pos_ct

        # Convert lists of markers to arrays
        markers1 = np.array(markers1)
        markers2 = np.array(markers2)

        # Center the markers at the origin
        markers1_mean = np.mean(markers1, axis=0)
        markers2_mean = np.mean(markers2, axis=0)
        markers1 -= markers1_mean
        markers2 -= markers2_mean

        # Calculate the cross-covariance matrix
        cross_cov = np.dot(markers1.T, markers2)

        # Calculate the singular value decomposition
        U, S, V_T = np.linalg.svd(cross_cov)

        # Calculate the rotation matrix
        R = np.dot(U, V_T)
        self.R = R

        # Check for reflection
        if np.linalg.det(R) < 0:
            V_T[2, :] *= -1
            R = np.dot(U, V_T)

        # Calculate the translation vector
        t = markers1_mean - np.dot(markers2_mean, R)
        self.t = t

        # Concatenate the rotation and translation matrices
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t
        """
        This matrix points FROM def TO ct
        I checked and can cofirm.
        """
        self.t_ct_def = transformation_matrix
        self.get_inverse()
        return transformation_matrix

    def get_inverse(self):
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self.R.T
        transformation_matrix[:3, 3] = - self.R.T @ self.t
        self.t_def_ct = transformation_matrix


# %%

class TrackerHandler(object):
    """
    This class should get a dicitonary of all tracker-paths
    then load the trackers and calculate the transformations based on the tracker list
    We have 5 trackers: we assume the are named and sorted as follows:
        -> Tracker 0: zf_mcp  -> trackers_def/{name}.csv -> trackers_ct/{name}.mrk.json
        -> Tracker 1: zf_dip  -> ""
        -> Tracker 2: dau_mcp -> ""
        -> Tracker 3: dau_dip -> ""
        -> Tracker 4: ft      -> ""
    """

    def __init__(self, path='../', def_path='trackers_def', ct_path='trackers_ct') -> None:
        """load all trackers and calculate their basic transformations"""
        def_path = f'{path}/{def_path}'
        ct_path = f'{path}/{ct_path}'

        # assign positions of the trackers in obshandler
        self.pos_zfmcp = 0
        self.pos_zfdip = 1
        self.pos_daumcp = 2
        self.pos_daudip = 3
        self.pos_ft = 4

        # initialise all trackers
        # index
        self.zf_mcp = Tracker(f'{def_path}/zf_mcp.csv',
                              f'{ct_path}/zf_mcp.mrk.json')
        self.zf_dip = Tracker(f'{def_path}/zf_dip.csv',
                              f'{ct_path}/zf_dip.mrk.json')

        # thumb
        self.dau_mcp = Tracker(
            f'{def_path}/dau_mcp.csv', f'{ct_path}/dau_mcp.mrk.json')
        self.dau_dip = Tracker(
            f'{def_path}/dau_dip.csv', f'{ct_path}/dau_dip.mrk.json')

        # ft
        self.ft = Tracker(f'{def_path}/ft.csv', f'{ct_path}/ft.mrk.json')

        self.t_ct_zfmcp = self.zf_mcp.t_ct_tr
        self.t_ct_zfdip = self.zf_dip.t_ct_tr
        self.t_ct_daumcp = self.dau_mcp.t_ct_tr
        self.t_ct_daudip = self.dau_dip.t_ct_tr
        self.t_ct_ft = self.ft.t_ct_tr

    def __call__(self, obs_handler: ObservationHandler):
        """
        Idea: take the data from the observationhandler and assign the current observation-data to the sensors
        -> each object should have a transformation, that shows the path FROM the Tracker TO the Base Tracker.
        The Base Tracker is ZF_MCP
        Example:
        T_CT_DAUDP = T_CT_ZFMCP * T_ZFMCP_OPT(t) * T_OPT_DAUDP(t)
        """

        # get the current transformations:
        t_opt_zfmcp = obs_handler.rigid_bodies[self.pos_zfmcp]._tmat
        t_opt_zfdip = obs_handler.rigid_bodies[self.pos_zfdip]._tmat
        t_opt_daumcp = obs_handler.rigid_bodies[self.pos_daumcp]._tmat
        t_opt_daudip = obs_handler.rigid_bodies[self.pos_daudip]._tmat
        t_opt_ft = obs_handler.rigid_bodies[self.pos_ft]._tmat

        # precalculate some transformations for better notation -> how to go from opt to ct?
        t_zfmcp_opt = np.linalg.inv(t_opt_zfmcp)
        t_ct_zfmcp = self.zf_mcp.t_ct_tr
        t_ct_opt = t_ct_zfmcp @ t_zfmcp_opt

        # calculate all required transformations
        self.t_ct_zfmcp = self.zf_mcp.t_ct_tr
        self.t_ct_zfdip = t_ct_opt @ t_opt_zfdip
        self.t_ct_daumcp = t_ct_opt @ t_opt_daumcp
        self.t_ct_daudip = t_ct_opt @ t_opt_daudip
        self.t_ct_ft = t_ct_opt @ t_opt_ft


# %%
if __name__ == '__main__':
    tr = Tracker(0, './Data/Trackers/DAU_DIP.csv')


# %%


def read_markerdata(path='/home/robotlab/Documents/GitHub/MA_Schote/MA/Data/test_01_31/Take 2023-01-31 06.07.05 PM.csv', body_name='DAU DIP', bodyt_markerf=True):
    """read the data of the specific body from the csv file
    -> get position for marker; -> get position + orientation for rigid body
    """
    with open(path, 'r') as file:
        reader = csv.reader(file)

        # skip headers
        next(reader)  # Skip the first row (header)
        next(reader)  # Skip the second row (empty row)

        # data info
        next(reader)  # marker types
        body_names = next(reader)  # name of rigid bodies
        id_names = next(reader)  # ids
        next(reader)  # position / rotation info
        next(reader)  # qx, qy, qz, qw, x, y, z -> info

        if bodyt_markerf and body_name in body_names:
            body_id = body_names.index(body_name)
            print(body_id)
        else:
            print('Name not found!')
            return

        if bodyt_markerf == False and body_name in id_names: # durch bodyt_maerkerf wird eine der beiden if-Bedingungen immer falsch sein.
            body_id = body_names.index(id_names)
        else:
            print('ID not found!')
            #return # Diese Zeile verhindert, dass man bei nutzunge eines Rigid Bodies weiter kommt.

        take_len = 7 if bodyt_markerf else 3

        # add data to result
        result = {
            'pos': [],
            'quat': [],
        }

        for row in reader:
            cur_res = row[body_id:body_id+take_len]

            if bodyt_markerf:
                result['quat'].append(cur_res[:4])
                result['pos'].append(cur_res[4:])
            else:
                result['pos'].append(cur_res)

        return result

# %%
