#%% import
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
import csv
import json
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import distance

#%% transformation optitrack tracker to real tracker
path_csv = "Data"
ct_folder = "Slicer3D"

def get_opti_positions(filename):
    '''Loads tracker information of a given tracker export file.'''
    path = path_csv + "/" + filename
    opti_positions = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == 'Name':
                name = row[1]
            if row[0] == 'Point':
                opti_positions.append([])
                for i in range(2,5):
                    opti_positions[-1].append(float(row[i]))

    #return name, opti_positions
    return opti_positions

def csv_test_load(testrun_path, tracker_designation_motive):
    '''This function is suppose to read in the trakcing data of a single tracker
    of a specified testrun from an motive .csv export.'''
    df = pd.read_csv(testrun_path, header=2, low_memory=False)
    start_coloum = df.columns.get_loc(tracker_designation_motive)
    data = df.iloc[3:,start_coloum:start_coloum+8]
    
    return data

def marker_variable_id(testrun_path, initialID=None, dtype="csv"):
    if dtype == "json":
        print("unable to load from json yet.")
        #df = load_marker_from_json(testrun_path, initalID)
        current_tracker_data = 0
        iloc_next_signal = 0
        df = pd.DataFrame()

    else:
        initialID = str(initialID)
        df = pd.read_csv(testrun_path, header=2, low_memory=False)
        iloc_next_signal = df.columns.get_loc(initialID)

        """Man will nur rechts des intialen Markers, nach Folgemarkern suchen, 
        da die ID der Marker in Motive immer steigend ist für neue Marker"""
        #df = df[:,start_coloum:]
        current_tracker_data = df.iloc[3:,iloc_next_signal:iloc_next_signal+3]
    
    # initialize variables
    added_data = np.zeros((current_tracker_data.shape))
    ID_end:int = 3; old_ID_end:int = 0
    next_line = 3
    dif = []

    
    while ID_end < df.index.stop:

        empty_cells = np.where(pd.isnull(df.iloc[next_line:,iloc_next_signal:iloc_next_signal+3]))
        ID_end = ID_end + int(empty_cells[0][0]) + 1
        #print("ID_end:", ID_end)
        print("dif =", ID_end-old_ID_end)
        dif.append(ID_end-old_ID_end)
        search_data = df.iloc[ID_end:,iloc_next_signal:]

        # step one: follow initialID until signal ends
        added_data[old_ID_end:ID_end,:] = current_tracker_data.iloc[old_ID_end:ID_end,:]
        old_ID_end = ID_end

        # find next signal from last timestemp, which is closest to 
        last_signal = current_tracker_data.iloc[ID_end-1,:]
        last_signal = np.array(list(map(float, last_signal)))
        min_dis = np.inf
        
        # in search data eins runter und eins rein, 
        # wegen zeilennummerierung, und da letztes value aus nan besteht
        # Zeilen
        for i, value in enumerate(search_data.values[1:,1:],start=0):
                # nur die nächsten x Zeitschritte 
                # nach nahe gelegenem Wert durchsuchen
                if i > 500:
                    break

                value = np.array(list(map(float, value)))
                # Spalten
                for j in range(0,len(value),3):
                    if np.isnan(value[j]) or np.isnan(value[j+1]) or np.isnan(value[j+2]):
                        continue
                    #print("value:", value[j:j+3])
                    #print("i:", i)
                    current_dis = np.absolute(np.linalg.norm(last_signal) - np.linalg.norm(value[j:j+3]))
                    if current_dis < min_dis:
                        iloc_next_signal = j
                        min_dis = current_dis
                        next_line = i
        current_tracker_data = df.iloc[ID_end+1:,iloc_next_signal:iloc_next_signal+3]

    return added_data
	
def plot_ply(tracker_points, opti_points, line_1, line_2, line_3, line_4):
    n = len(tracker_points)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    x, y, z = [], [], []
    x2, y2, z2 = [], [], []
    for i in range(0,n):
            x.append(tracker_points[i][0])
            y.append(tracker_points[i][1])
            z.append(tracker_points[i][2])
    ax.scatter(x,y,z,c='b', marker='^')
    #print('x:')
    #print(x)
    for i in range(0,n):
            x2.append(opti_points[i][0])
            y2.append(opti_points[i][1])
            z2.append(opti_points[i][2])
    #print('x2:')
    #print(x2)
    ax.scatter(x2,y2,z2, c='r', marker='o')
    ax.plot([line_1[0][0],line_1[1][0]],[line_1[0][1],line_1[1][1]], zs=[line_1[0][2],line_1[1][2]], c='b')
    ax.plot([line_2[0][0],line_2[1][0]],[line_2[0][1],line_2[1][1]], zs=[line_2[0][2],line_2[1][2]], c='b')
    ax.plot([line_3[0][0],line_3[1][0]],[line_3[0][1],line_3[1][1]], zs=[line_3[0][2],line_3[1][2]], c='r')
    ax.plot([line_4[0][0],line_4[1][0]],[line_4[0][1],line_4[1][1]], zs=[line_4[0][2],line_4[1][2]], c='r')

    plt.show()


def get_min_max_dis(points):
    n = len(points)
    d_comp_max = 0
    d_comp_min = math.inf
    for i in range(n):
        print('i:', i)
        for j in range(n-1,-1+i,-1):
            if j != i:
                print('    j:', j)
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                dz = points[j][2] - points[i][2]
                
                d_diff= math.sqrt(dx**2+dy**2+dz**2)
        
                if d_diff > d_comp_max:
                    v_max = [points[j], points[i]]
                    d_comp_max = d_diff
                    print('v_max:', v_max, 'mit d =', d_comp_max)
                if d_diff < d_comp_min:
                    v_min = [points[j], points[i]]
                    d_comp_min = d_diff
                    print('v_min:', v_min, 'mit d =', d_comp_min)
    return v_max, v_min

def sort_points_by_dis(points):
    n = len(points)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            d_diff = distance.euclidean(points[i],points[j])
            pairs.append([points[i], points[j], d_diff])
    pairs.sort(key=lambda x: x[2])
    return pairs

def compare_point_lists(pairs1, points1, pairs2, points2):
    '''Lese Punktepaare, und Punktewolken ein. Vergleiche Positionen der Punkte anhand
    der Stellen in den Punktepaaren, in denen sie
    vorkommen.'''
    distance_value_in_points1 = [[] for _ in range(len(points1))]
    distance_value_in_points2 = [[] for _ in range(len(points2))]

    '''Erstelle für jeden Punkt eine Liste, in der Steht in welcher Distanz er vokommt.'''
    for i in range(len(pairs1)):
        for j in range(len(points1)):
            if points1[j] in pairs1[i]:
                distance_value_in_points1[j].append(i + 1)
            if points2[j] in pairs2[i]:
                distance_value_in_points2[j].append(i + 1)
    print(distance_value_in_points1)
    print(distance_value_in_points2)

    '''An dieser Stelle soll die Liste der Punkte aus dem CT (2) anhand der Punkte aus dem Opti-Export (1)
    sortiert werden. Dafür wird der Index einer Distanz Index Kombi von (2) in (1) gesucht und der Index gepseichert.
    Anhand der entstehenden Liste von Indexen werden die Punkte von (2) umsortiert.'''''
    index_list = []
    for i in range(len(points1)):
        index_list.append(distance_value_in_points1.index(distance_value_in_points2[i]))

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    points_2_out = [x for _, x in sorted(zip(index_list, points2))]

    return points1, points_2_out
# %% 
def calculate_transformation_matrix(markers1, markers2):
    '''Setzt vorraus, dass die Punktelisten korrekt sortiert sind.'''
    # Convert lists of markers to arrays
    markers1 = np.array(markers1, dtype=float)
    markers2 = np.array(markers2, dtype=float)

    # Calculate the rotation matrix, R transforms b to a.
    R = Rot.align_vectors(markers2,markers1)

    # Calculate the translation vector
    t = np.mean(markers2, axis=0) - np.dot(np.mean(markers1, axis=0), R)
    
    # Concatenate the rotation and translation matrices
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    return transformation_matrix
# %%
def Rotation_Matrix(phi, theta, psi, degrees = False):
    '''Gibt Rotationsmatrix für Eulerwinkel zurück.'''
    if phi > 2*math.pi or theta > 2*math.pi or \
        psi > 2*math.pi or degrees == True:
        print('Calculating R for Input in Degrees and Euler ZYX.')
        phi = phi * 180 / math.pi
        theta = theta * 180 / math.pi
        psi = psi * 180 / math.pi
    else: 
        print('Calculating R for Input in Radians and Euler ZYX.')
    
    Rx = np.array([[1, 0, 0], \
        [0, math.cos(phi), -math.sin(phi)], \
        [0, math.sin(phi), math.cos(phi)]])
    
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0], \
        [math.sin(psi), math.cos(psi), 0], \
        [0, 0, 1]])
    
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)], \
        [0, 1, 0], \
        [-math.sin(theta), 0, math.cos(theta)]])

    R_zw = np.matmul(Rz,Ry)
    R = np.matmul(R_zw,Rx)
    return R

def min_max_arrays_to_kosy(min_track, max_track):
    x = np.linalg.norm(max_track[0]+min_track[1]-min_track[0])
    z = np.linalg.norm(max_track[1]-max_track[0])
    y = np.cross(x,z)

    kosy = [x,y,z]

    return kosy
    



#%% Function tests
if __name__ == '__main__':
    path = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'
    #raw_data = csv_test_load(path, '55')
    marker_data = marker_variable_id(path, 'Unlabeled 2016')
 
# %%
#   class Tracker_3dicke:
 #       numTrackers = 5
 #       positions = [[0, 0, 75], [-42, 0, 46], [25, 0, 46], [0, 37, 41.5], [0, -44, 41.5]] # [[x,y,z],[x2,y2,z2],...]
 #       name, opti_positions = get_opti_positions('MakerJS_3dicke.csv')
#
#    class Tracker_Nico:
#        numTrackers = 5
#        positions = [[0, 0, 61], [-41, 0, 35], [20, 0, 35], [-10, 31, 35], [-10, -14, 35]] # [[x,y,z],[x2,y2,z2],...]
#        name, opti_positions = get_opti_positions('Tracker Nico.csv')