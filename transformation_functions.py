#%% import
import os
from datetime import datetime
import trackers
import pandas as pd
import numpy as np
import stl
import math
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
import csv
import json
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import distance
from scipy.signal import butter, lfilter, freqz, buttord, filtfilt
from tqdm import tqdm

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
        next_col = 0
        df = pd.DataFrame()

    else:
        initialID = str(initialID)
        df = pd.read_csv(testrun_path, header=2, low_memory=False)
        next_col = df.columns.get_loc(initialID)

        """Man will nur rechts des intialen Markers, nach Folgemarkern suchen, 
        da die ID der Marker in Motive immer steigend ist für neue Marker"""
        #df = df[:,start_coloum:]
        
    # initialize variables
    j_safe = 0
    i_safe = 0
    next_line = 3
    
    next_line_old = 3
    dif = []
    current_tracker_data = df.iloc[next_line_old:,next_col:next_col+3]
    added_data = np.zeros((current_tracker_data.shape))

    filled_cells = np.where(pd.notna(current_tracker_data.iloc[next_line:,:]))
    next_line = int(filled_cells[0].max())
    
    for k in tqdm(range(df.index.stop)):
    #while ID_end <= df.index.stop-3:
        if next_line == df.index.stop:
            break

        #empty_cells = np.where(pd.isnull(df.iloc[next_line:,next_col:next_col+3]))
        filled_cells = np.where(pd.notna(current_tracker_data.iloc[next_line:,:]))
        # was wenn der nächste Marker mit ein paar Nans beginnt?
        #next_line = int(filled_cells[0].max())

        if next_line < next_line_old:
            print("next line < old next line")
            break

        dif.append(next_line-next_line_old)
        search_data = df.values[next_line+1:,next_col+3:]

        # step one: follow initialID until signal ends
        # save data in added_data for output
        added_data[next_line_old:next_line,:] = current_tracker_data.iloc[next_line_old:next_line,:]
        print('writing in:', next_line_old, 'to', next_line)

        # find next signal from last timestemp, which is closest to 
        last_signal = current_tracker_data.iloc[next_line-2, 0:3]
        last_signal = np.array(list(map(float, last_signal)))

        if last_signal[0] == np.nan:
            print("last signal:", last_signal, last_signal.shape)
            break

        min_dis = np.inf
        current_dis = np.inf
        
        # in search data eins runter und eins rein, 
        # wegen zeilennummerierung, und da letztes value aus nan besteht
        # Zeilen
        for i, value in enumerate(search_data,start=0):
            # nur die nächsten x Zeitschritte 
            # nach nahe gelegenem Wert durchsuchen
            if i >= 4000:
                break

            value = np.array(list(map(float, value)))

            # Spalten
            for j in range(0,len(value),3):

                if np.isnan(value[j]) or np.isnan(value[j+1]) or np.isnan(value[j+2]):
                    continue
                
                #print("value:", value[j:j+3])
                #print("i:", i)
                else:
                    current_dis = np.absolute(np.linalg.norm(last_signal) - np.linalg.norm(value[j:j+3]))
                    
                    if current_dis < min_dis:
                        j_safe = j
                        min_dis = current_dis
                        i_safe = i
                        #print("next line:", next_line)
        
        next_col += j_safe
        next_line_old = next_line + 1
        next_line += i_safe

        if current_dis == np.inf:
            print("no marker nearby!")
            break
        if next_col > df.shape[1] or next_line > df.shape[0]:
            print("end of dataframe reached!")
            break

        current_tracker_data = df.iloc[3:,next_col:next_col+3]
        
        #ID_end = ID_end + int(empty_cells[0][0]) + 1

    return added_data

def marker_variable_id_linewise(testrun_path, initialID=None, dtype="csv", d_max = 56):
    if dtype == "json":
        print("unable to load from json yet.")
        #df = load_marker_from_json(testrun_path, initalID)
        next_col = 0
        df = pd.DataFrame()

    else:
        initialID = str(initialID)
        df = pd.read_csv(testrun_path, header=2, low_memory=False)
        next_col = df.columns.get_loc(initialID)

    # variablen initiieren.
    start_line = 3
    dis_list = []
    start_value = df.values[start_line,next_col:next_col+3]

    """If condition to be able to catch trackers, that a not visiable immidiatly"""
    if np.isnan(float(start_value[0])):
        filled_cells = np.where(pd.notna(df.iloc[:,next_col]))
        start_line = int(filled_cells[0][3])
        start_value = df.values[start_line,next_col:next_col+3]

    added_data = np.zeros((df.shape[0]-start_line,3)) 
    added_data[0,:] = start_value
    last_signal = added_data[0,:]

    # Start Zeilenschleife
    for k in tqdm(range(1,added_data.shape[0])):
        
        min_dis = np.inf
        current_dis = np.inf

        value = df.values[k+3,:]
        value = np.array(list(map(float, value)))

        # Spalten
        values_to_add = [np.nan, np.nan, np.nan]

        for j in range(next_col,len(value),3):

            #print(df.iloc[0,j:j+3])
            #Unlabeled 2291      E0C50
            #Unlabeled 2291.1    E0C50
            #Unlabeled 2291.2    E0C50
            
            if np.isnan(value[j]) or np.isnan(value[j+1]) or np.isnan(value[j+2]):
                continue            

            else:
                current_dis = np.absolute(np.linalg.norm(last_signal) - np.linalg.norm(value[j:j+3]))
                
                if current_dis < min_dis:
                    min_dis = current_dis
                    values_to_add = value[j:j+3]

        # safe closest values from line k
        if values_to_add[0] == np.nan:
            continue
        else:
            last_signal == values_to_add
        
        # falls die Minimale Distanz zum nächsten Punkt den Grenzwert überschreitet,
        # wird der Punkt nicht in die Ausgabeliste eingetragen.
        dis_list.append(min_dis)
        if min_dis >= d_max:
            added_data[k,:] = [np.nan, np.nan, np.nan]
        else:
            
            added_data[k,:] = values_to_add

    #print(dis_list)
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

'''Tiefpassfilter'''
def butter_lowpass(wn, fs, order=5):
    b, a = butter(order, wn, btype='low', analog=False, fs=fs)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis = 0)
    return y

def plot_tiefpass(fs, Gp, Gs, wp, ws, marker_data):
    order, wn = buttord(wp, ws, Gp, Gs)
    y = butter_lowpass_filter(marker_data, wn, fs, order)
    
    # Plotting
    fig = plt.subplot(1, 1, 1)
    fig.plot(marker_data, 'b-', linewidth=0.25, label='marker data')

    fig.plot(y, 'r-', linewidth=0.5, label='filtered data')
    plt.xlabel('Time [120 Hz]')
    plt.title("Marker")
    fig.grid()
    fig.legend()

    plt.subplots_adjust(hspace=0.35)
    now = datetime.now()
    plot_file_title = "marker_" + now.strftime("%d_%m_%Y_%H_%M_%S")
    # plt.savefig(plot_file_title + ".pdf", format="pdf")
    return y
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
# calculate angle beween two vectors
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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
    
def nan_helper(a):
    x, y = np.indices(a.shape)
    interp = np.array(a)
    interp[np.isnan(interp)] = interpolate.griddata((x[~np.isnan(a)], y[~np.isnan(a)]), a[~np.isnan(a)], (x[np.isnan(a)], y[np.isnan(a)])) 
    return interp

def read_markups(path):
    """read the data of the marker points from the ct scan"""
    # 1. load mounting points
    with open(path) as jsonfile:
        data = json.load(jsonfile)

    # extract point infos
    point_data = data['markups'][0]['controlPoints']
    point_list = [point['position'] for point in point_data]
    return point_list
'''Hilfspunkte 1-4 an Daumen.
	1 mittig-distal
	2 Distal
	3 mittig- proximal
	4 proximal
	
Hilfspunkte 5-9 an Zeigefinger (ZF)
	8, 9: Kontakpunkt distal-mittig
	7 Kontaktpunkt mittig-proximal
	5 proximal
    6 mittig-proximal'''

def t_cog_trackerorigin(tracker_origin = np.array([0,0,0]), cog = np.array([0,0,0])):
    t = np.array([0,0,0])
    t = np.subtract(cog,tracker_origin)
    return t

def get_helper_points(helper_ids = [1,2,3,4,5,6,7,8,9], path = './Slicer3D/Hilfspunkte.mrk.json'):
    # change format of helper_ids to string
    helper_ids = [str(i) for i in helper_ids]
    
    helper_points = []

    # 1. load mounting points
    with open(path) as jsonfile:
        data = json.load(jsonfile)

    # extract point infos
    point_data = data['markups'][0]['controlPoints']
    helper_points = [point['position'] for point in point_data if point['id'] in helper_ids]
    return helper_points

class bone_stl(trackers.Tracker):
    
    def __init__(self, folder_path = "./Data/STL", finger_name = "") -> None:
        super().__init__(0, './Data/Trackers/'  + finger_name + '.csv')
        # Setting standard filter variables.
        fs = 120.0
        Gp = 0.1
        Gs = 3.0
        wp = 0.8
        ws = 1.1
        
        #tr = trackers.Tracker.__init__(self, 0, './Data/Trackers/DAU_DIP.csv')
        
        for file in os.listdir(folder_path):
            if file.find(finger_name) >= 0: 
                # print(os.path.join(directory, filename))
                file_path = os.path.join(folder_path, file)
                stl_data = stl.mesh.Mesh.from_file(file_path)
            else:
                continue
        self.volume, self.cog_stl, self.inertia = stl_data.get_mass_properties()
        self.t_tracker_CT = np.subtract(self.cog_stl, self.marker_pos_ct)
        self.d_tracker_CT = np.linalg.norm(self.t_tracker_CT)
        
        helper_ids = []
        if finger_name == "DAU_DIP":
            helper_ids = [2,1]
            marker_ID = 'Unlabeled 2016'
        elif finger_name == "DAU_MCP":
            helper_ids = [1,3]
            marker_ID = 'Unlabeled 2016'
        elif finger_name == "ZF_DIP":
            helper_ids = [8,9]
            marker_ID = 'Unlabeled 2016'
        elif finger_name == "ZF_MCP":
            helper_ids = [7,5]
            marker_ID = 'Unlabeled 2016'

        # helper points
        self.helper_points = get_helper_points(helper_ids)

        self.t_proxi_CT = t_cog_trackerorigin(self.helper_points[0], self.marker_pos_ct)
        self.d_proxi_CT = np.linalg.norm(self.t_tracker_CT)

        self.t_dist_CT = t_cog_trackerorigin(self.helper_points[1], self.marker_pos_ct)
        self.d_dist_CT = np.linalg.norm(self.t_tracker_CT)

        # build marker trace from csv file
        marker_trace = marker_variable_id_linewise(self.path, marker_ID, "csv", 40)
        inter_data = nan_helper(marker_trace)
        self.opti_marker_trace = plot_tiefpass(fs, Gp, Gs, wp, ws, inter_data)

# %%

#%% Function tests
if __name__ == '__main__':

    ZF_DIP = bone_stl(finger_name="ZF_DIP")

    #path = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'
    path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

    #raw_data = csv_test_load(path, '55')
    marker_ID = 'Unlabeled 2016'
    marker_data = marker_variable_id_linewise(path, marker_ID, "csv", 40)
    inter_data = nan_helper(marker_data)
    # Setting standard filter variables.
    fs = 120.0
    Gp = 0.1
    Gs = 3.0
    wp = 0.8
    ws = 1.1

    pic = plot_tiefpass(fs, Gp, Gs, wp, ws, inter_data)
    filtered_data = interact(plot_tiefpass, fs = fixed(fs), Gp = widgets.FloatSlider(value=Gp, min=0,max=2,step=0.1),
                                      Gs = widgets.FloatSlider(value=Gs, min=0,max=120,step=1), 
                                      wp = widgets.FloatSlider(value=wp, min=0,max=2,step=0.05), 
                                      ws = widgets.FloatSlider(value=ws, min=0,max=2,step=0.05),
                                        marker_data = fixed(inter_data))
    

# %%
#   Tracker_3dicke:
 #       numTrackers = 5
 #       positions = [[0, 0, 75], [-42, 0, 46], [25, 0, 46], [0, 37, 41.5], [0, -44, 41.5]] # [[x,y,z],[x2,y2,z2],...]
 #       name, opti_positions = get_opti_positions('MakerJS_3dicke.csv')
#
#    Tracker_Nico:
#        numTrackers = 5
#        positions = [[0, 0, 61], [-41, 0, 35], [20, 0, 35], [-10, 31, 35], [-10, -14, 35]] # [[x,y,z],[x2,y2,z2],...]
#        name, opti_positions = get_opti_positions('Tracker Nico.csv')