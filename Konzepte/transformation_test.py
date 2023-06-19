#%% Import packages
import sys
import numpy as np
sys.path.append('./mujoco')
sys.path.append('./Konzepte')
import transformation_functions as tf
import Konzepte.trackers as trackers
import my_write_parameters as mwp
import my_model as mwj
import yaml
from pyquaternion import Quaternion
import importlib
importlib.reload(tf)
importlib.reload(trackers)
from scipy.spatial.transform import Rotation as scipyR


def kabsch_iter(points1,points2,iterations=5):
        '''marker_pos_def = points2'''
        rssd = np.inf
        T = Tracker_DAU_DIP.calculate_transformation_matrix(points1, points2)
        marker_pos_new = [np.append(pos,1) for pos in points2]
        for k in range(len(points2)):
            marker_pos_new[k] = np.matmul(T, np.array(marker_pos_new[k]))
            marker_pos_new[k] = marker_pos_new[k][:3]
        
        for i in range(iterations):
            T_next = scipyR.from_matrix(np.eye(3))
            T_next = scipyR.align_vectors(points1, marker_pos_new)
            for j in range(len(marker_pos_new)):
                marker_pos_new[j] = T_next.as_matrix() @ np.array(marker_pos_new[j]) # type: ignore
            R = np.matmul(T_next.as_matrix(), T[:3,:3]) # type: ignore
            T[:3,:3] = R
            print(rssd)
        return T

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

def test_kabsch():
    """Test the Kabsch algorithm function with a simple rotation and translation."""
    # Define a set of points P
    P = np.array([[131.4972845626892, -197.37202996092802, -134.52503590000154],
                [105.20596849999995, -187.66414017556622, -156.4512927417677],
                [105.20596849999995, -191.5862107698484, -123.76737112274945],
                [118.99568454341468, -219.3519672497178, -120.1142180000017],
                [111.3966727669929, -227.11439863423468, -159.66018340000167]])
    # Define a rotation matrix
    angle = np.radians(30)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])
    # Define a translation vector
    t = np.array([10, 20, 30])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    P_homogenous = np.c_[P, np.ones(len(P))]
    Q_homogenous = np.dot(P_homogenous, T.T)
    Q = Q_homogenous[:, :3]
    # Use the Kabsch algorithm to compute the transformation from Q to P
    T_estimated  = kabsch(P, Q)
    # Verify the result
    assert np.allclose(T_estimated [:3, :3], R, atol=1e-6), "Rotation matrix is incorrect"
    print(np.round(T_estimated, 3))
    print(np.round(T, 3))

#%% Definition der Pfade
if __name__ == '__main__':
    test_metadata = tf.get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
    hand_metadata = tf.get_json('hand_metadata.json')
    data_path = 'Data/test_01_31/'
    test_file = '2023_01_31_18_12_48.json'
    opt_data = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

    #%% Laden der Klassen
    Tracker_DAU_DIP = tf.tracker_bone('DAU_DIP',test_path=test_metadata['path'])
    Tracker_DAU_MCP = tf.tracker_bone('DAU_MCP',test_path=test_metadata['path'])
    Tracker_ZF_midhand = tf.tracker_bone('ZF_midhand',test_path=test_metadata['path'])

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
    # %% Test

    marker_pos_new = [np.append(pos,1) for pos in Tracker_DAU_DIP.marker_pos_ct]
    for k in range(len(Tracker_DAU_DIP.marker_pos_def)):
        marker_pos_new[k] = np.dot(Tracker_DAU_DIP.T_ct_def, np.array(marker_pos_new[k]))
        marker_pos_new[k] = marker_pos_new[k][:3]
    print(marker_pos_new)
    print(Tracker_DAU_DIP.marker_pos_def)

    # %%


    T5 = kabsch_iter(Tracker_DAU_DIP.marker_pos_ct, Tracker_DAU_DIP.marker_pos_def)
    # %% Test des Translationsvektors t mit Kabsch von tracker.Tracker.transformation_matrix
    

    alpha = 0.5
    R = np.array([[np.cos(alpha), -np.sin(alpha), 0, 1],
                    [np.sin(alpha), np.cos(alpha), 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1]])

    P = np.array(Tracker_DAU_DIP.marker_pos_def)
    Q = np.array(Tracker_DAU_DIP.marker_pos_ct)

    #Q = np.zeros((5,3))
    #for i in range(5):
    #    Q[i,:] = np.matmul(R, np.append(P[i],1))[:3]

    T_ct_def = Tracker_DAU_DIP.calculate_transformation_matrix(P,Q) # P nach Q
    T_def_ct = Tracker_DAU_DIP.invert_T(T_ct_def) # Q nach P

    P_recover = np.zeros((5,3))
    for i in range(5):
        P_recover[i,:] = np.matmul(T_def_ct, np.append(Q[i,:],1))[:3]
        print(np.round(P_recover[i,:],4))
        print(np.round(P[i],4))
        print('---')

    test_kabsch()


    points1 = [[12,12,12],[4,4,4],[5,5,5], [7,7,7]]
    points2 = [[0,0,0],[4,4,4], [6,6,6], [11,11,11]]

    points1, points2 = tf.sort_points_relative(points1, points2)

    print(points1)
    print(points2)


