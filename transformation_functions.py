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
import matplotlib.colors as mcolors
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
import csv
import codecs, json 
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
    data = df.values[3:,start_coloum:start_coloum+8]
    data = np.array([list(map(float, i)) for i in data])
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
        print("start value:", start_value)
        print("start line:", start_line)

    #added_data = np.zeros((df.shape[0]-start_line,3)) 
    added_data = np.zeros((df.shape[0]-3,3)) 
    added_data[0,:] = start_value
    last_signal = added_data[0,:]

    # Start Zeilenschleife
    #for k in tqdm(range(1,added_data.shape[0])):
    for k in tqdm(range(start_line,added_data.shape[0])):
        
        min_dis = np.inf
        current_dis = np.inf

        #value = df.values[k+3,:]
        value = df.values[k,:]
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
    print("len added_data for marker %s:" % initialID, len(added_data))
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

def plot_class(i, Trackers1: list = [], Trackers2: list = [], names: list = [], radius: list = [], save = False, show = True):
    '''Plot tracker points in 3D for timestep i with different colors and a sphere with radius d around each point'''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    morecolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    longcolors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    css4colors = list(mcolors.CSS4_COLORS.values())
    customcolors = longcolors + morecolors + ['darkblue', 'darkgreen', 'darkred', 'darkcyan', 'darkmagenta', 'darkyellow', 'darkgray', 'darkolive', 'darkcyan']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    # Plot known tracker points
    for j in range(len(Trackers1)):
        ax.scatter(Trackers1[j][i][0],Trackers1[j][i][1],Trackers1[j][i][2], c=customcolors[j], marker=markers[0], label=names[j])
    for k in range(len(Trackers2)):
        ax.scatter(Trackers2[k][i][0],Trackers2[k][i][1],Trackers2[k][i][2], c=customcolors[j+k+1], marker=markers[2], label=names[j+k+1])
    
    # Plot lines between known tracker points
    for j in range(len(Trackers1)-1):
        ax.plot([Trackers1[j][i][0],Trackers1[j+1][i][0]],[Trackers1[j][i][1],Trackers1[j+1][i][1]], zs=[Trackers1[j][i][2],Trackers1[j+1][i][2]], c='r')
    for k in range(len(Trackers2)-1):
        ax.plot([Trackers2[k][i][0],Trackers2[k+1][i][0]],[Trackers2[k][i][1],Trackers2[k+1][i][1]], zs=[Trackers2[k][i][2],Trackers2[k+1][i][2]], c='b')
    
    # Plot sphere around known tracker points
    for j in range(len(Trackers1)):
        d = radius[j]
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = Trackers1[j][i][0] + d*np.cos(u)*np.sin(v)
        y = Trackers1[j][i][1] + d*np.sin(u)*np.sin(v)
        z = Trackers1[j][i][2] + d*np.cos(v)
        ax.plot_wireframe(x, y, z, color='r', alpha=0.1)
    for k in range(len(Trackers2)):
        d = radius[j+k+1]
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = Trackers2[k][i][0] + d*np.cos(u)*np.sin(v)
        y = Trackers2[k][i][1] + d*np.sin(u)*np.sin(v)
        z = Trackers2[k][i][2] + d*np.cos(v)
        ax.plot_wireframe(x, y, z, color='b', alpha=0.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    ax.set_zlim(-250,250)
    ax.legend(loc="upper left",bbox_to_anchor=(1.1, 1), ncol = 2)
    if save:
        dt = datetime.now()
        dt = dt.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig('plots/plot_class_'+str(i)+'.pdf', dpi=600)
    if show:
        plt.show()
    plt.close()
# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def stl_find_mins_maxs(obj):
    '''Gives back the max dimensions in x, y and z direction of an stl object.'''
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz
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
    #fig.legend()

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

# https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    
    A quaternion rotation is made up of 4 numbers, 
    whose values all have a minimum of -1 and a maximum of 1, 
    i.e (0, 0, 0, 1) is a quaternion rotation that is equivalent to 
    'no rotation' or a rotation of 0 around all axis. But ouptut of this function
    is a 3x3 matrix: array([[-1,  0,  0],
                            [ 0, -1,  0],
                            [ 0,  0,  1]]) for Input [0,0,0,1]
    https://answers.unity.com/questions/645903/please-explain-quaternions.html#:~:text=A%20quaternion%20rotation%20is%20made,of%200%20around%20all%20axis.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

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
    interp[np.isnan(interp)] = interpolate.griddata((x[~np.isnan(a)], y[~np.isnan(a)]), a[~np.isnan(a)], (x[np.isnan(a)], y[np.isnan(a)]), method='nearest') 
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

def get_helper_points(finger_name: str, path = './Slicer3D/Joints/'):
    '''Returns the joint points for the finger_name.'''
    
    helper_points = {'DAU_DIP':[], 'DAU_MCP':[], 'DAU_PIP':[], 'ZF_DIP':[], 'ZF_MCP':[], 'ZF_PIP':[]}

    for key in helper_points.keys():
        file_path = path + key[:3] + '_A' + key[3:] + '.json'        # 1. load mounting points
        with open(file_path) as jsonfile:
            data = json.load(jsonfile)

        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        helper_points[key] = [point['position'] for point in point_data]
        helper_points[key].append(np.mean(helper_points[key], axis=0))
    return helper_points[finger_name]

def get_signle_joint_file(file_path = ['./Slicer3D/DAU_COG.mrk.json']):
    '''Returns the joint points for the finger_name.'''
    with open(file_path) as jsonfile:
        data = json.load(jsonfile)

    # extract point infos
    point_data = data['markups'][0]['controlPoints']
    helper_points = [point['position'] for point in point_data]
    return np.array(helper_points)

def get_joints(path = ['./Slicer3D/Joints/']):
    '''Returns the joint points for the finger_name.'''
    joint_pos = []

    for file_path in path:
        if file_path[-5:] != '.json':
            joint_pos.append(np.array([np.nan,np.nan,np.nan]))
        else:
            with open(file_path) as jsonfile:
                data = json.load(jsonfile)

            # extract point infos
            point_data = data['markups'][0]['controlPoints']
            helper_points = [point['position'] for point in point_data]
            joint_pos.append(np.mean(helper_points, axis=0))
    return joint_pos

def get_test_metadata(name):
    '''Returns the metadata of the test.'''
    with open('test_metadata.json') as json_data:
        d = json.load(json_data)
        metadata = d[name]
        json_data.close()
    return metadata
def get_json(path):
    '''Returns the metadata of the test.'''
    with open(path) as json_data:
        d = json.load(json_data)
        json_data.close()
    return d
class tracker_bone(trackers.Tracker):
    
    def __init__(self, finger_name = "", test_path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv') -> None:
        self.finger_name = finger_name
        print("Finger:", self.finger_name)
        self.metadata = self.get_metadata()
        if self.metadata["tracking"] == "Tracker":
            super().__init__(0, self.metadata['tracker def opti'], self.metadata['tracker def CT'], self.finger_name)
        else:
            print("No tracker data available.", self.metadata["tracking"])
        
        # get the information of the stl file
        stl_data = stl.mesh.Mesh.from_file(self.metadata['stl'])
        self.volume, self.cog_stl, self.inertia = stl_data.get_mass_properties()
        
        '''This Part is only for full trackers.'''
        if self.metadata["tracking"] == "Tracker":
            # Get the trajectory of the tracker from the test data
            self.track_traj_opti = csv_test_load(test_path, self.metadata['tracker name'])
            
            # initialize the transformation matrix
            self.T_opti_i = np.zeros((len(self.track_traj_opti),4,4))
            self.T_i_opti = np.zeros((len(self.track_traj_opti),4,4))
            self.T_ct_i = np.zeros((len(self.track_traj_opti),4,4))
            self.T_i_ct = np.zeros((len(self.track_traj_opti),4,4))

            
            # T from timestamp i to opti coordinate system
            for i in range(len(self.track_traj_opti)):
                R = Quaternion(self.track_traj_opti[i,:4]).rotation_matrix
                t = self.track_traj_opti[i,4:7]
                self.T_opti_i[i,:3,:3] = R
                self.T_opti_i[i,:3,3] = t
                self.T_opti_i[i,3,3] = 1
                #inverse T
                self.T_i_opti[i,:,:] = self.invert_T(self.T_opti_i[i,:,:])
                
                
                # calculate the trajectory of the tracker in the CT coordinate system
                self.T_ct_i[i,:,:] = self.T_ct_def @ self.T_opti_i[i,:,:]
                self.T_i_ct[i,:,:] = self.invert_T(self.T_ct_i[i,:,:])
            print(self.T_opti_i[0,:,:])

            
            # calculate the trajectory of the tracker in the CT coordinate system
            self.track_traj_CT = np.zeros((len(self.track_traj_opti),3))

            self.track_rot_CT = np.zeros((len(self.track_traj_opti),4))

            self.track_traj_CT = [np.matmul(self.T_ct_def[:3,:3],self.track_traj_opti[i,4:7]) + self.T_ct_def[3,:3] \
                                  for i in range(len(self.track_traj_opti))] #checked
            
            self.track_rot_CT = [Quaternion(self.track_traj_opti[i,:4]).rotate(Quaternion(matrix=self.T_ct_def[:3,:3])) \
                                    for i in range(len(self.track_traj_opti))] #checked

            # np.mean(self.marker_pos_ct,axis=0) ist hier anwendbar, da das mean der maker pos im def file bei [0,0,0] liegt.
            self.t_tracker_CT = np.subtract(np.mean(self.marker_pos_ct,axis=0), self.cog_stl)
            self.r_tracker_CT = self.t_tracker_CT
            self.d_tracker_CT = np.linalg.norm(self.t_tracker_CT)
            """cog_traj_CT[i] =  R_ct_opti * pos_track_opti[i] + R_ct_opti * opti_R[i] * r_rel_cog_tracker"""
            # position
            self.cog_traj_CT = [self.track_traj_CT[i] + self.track_rot_CT[i].rotate(self.t_tracker_CT) for i in range(len(self.track_traj_opti))] # checked
            # orientation
            self.cog_rot_CT = self.track_rot_CT # checked

            self.helper_points = get_joints(self.metadata['joints'])

            if not np.isnan(self.helper_points[0][0]):
                self.t_proxi_CT = np.subtract(self.helper_points[0], np.mean(self.marker_pos_ct,axis=0))
                self.d_proxi_CT = np.linalg.norm(self.t_proxi_CT)
                self.proxi_traj_CT = [np.matmul(self.T_ct_def[:3,:3],self.track_traj_opti[i,4:7]) + self.T_ct_def[3,:3] \
                                  + np.matmul(np.matmul(self.T_ct_def[:3,:3], Quaternion(self.track_traj_opti[i,:4]).rotation_matrix), self.t_proxi_CT) \
                                  for i in range(len(self.track_traj_opti))]
            else:
                print('No proximal joint found.')

            # Warum war diese Zeile eins ausgerückt? Als noch try und except drin war?
            if not np.isnan(self.helper_points[1][0]):
                self.t_dist_CT = np.subtract(self.helper_points[1], np.mean(self.marker_pos_ct,axis=0))
                self.d_dist_CT = np.linalg.norm(self.t_dist_CT)
                self.dist_traj_CT = [np.matmul(self.T_ct_def[:3,:3],self.track_traj_opti[i,4:7]) + self.T_ct_def[3,:3] \
                                  + np.matmul(np.matmul(self.T_ct_def[:3,:3], Quaternion(self.track_traj_opti[i,:4]).rotation_matrix), self.t_dist_CT) \
                                  for i in range(len(self.track_traj_opti))]
            else:
                print('No distal joint found.')


    def get_metadata(self):
        '''Returns the metadata of the Phalanx.'''
        with open('hand_metadata.json') as json_data:
            d = json.load(json_data)
            metadata = d[self.finger_name]
            json_data.close()
        return metadata
    
    def invert_T(self, T = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])):
        ''' gives back the inversese of a 4x4 transformation matrix'''
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = T[:3,:3].T
        transformation_matrix[:3, 3] = - T[:3,:3].T @ T[:3,3]
        return transformation_matrix
    
class marker_bone(tracker_bone):
    '''Class for the bones with markers on top. Inherits from tracker_bone.'''
    def __init__(self, finger_name = "", test_path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv', init_marker_ID = "Unlabeled ...") -> None:
        super().__init__(finger_name=finger_name, test_path=test_path)

        base_tracker = tracker_bone("ZF_midhand", test_path=test_path)
        self.testname = os.path.split(test_path)[1]
        test_metadata = get_test_metadata(self.testname)

        # get marker information
        self.marker_pos_ct = self.get_marker_pos_ct()
        self.joints = get_joints(self.metadata['joints'])

        self.t_proxi_CT = np.subtract(self.joints[0], self.marker_pos_ct)
        self.d_proxi_CT = np.linalg.norm(self.t_proxi_CT)
        
        self.t_dist_CT = np.subtract(self.joints[1], self.marker_pos_ct)
        self.d_dist_CT = np.linalg.norm(self.t_dist_CT)

        # get stl information
        stl_data = stl.mesh.Mesh.from_file(self.metadata['stl'])
        self.volume, self.cog_stl, self.inertia = stl_data.get_mass_properties()
        self.t_cog_CT = np.subtract(self.cog_stl, np.mean(self.marker_pos_ct,axis=0)) 
        self.d_cog_CT = np.linalg.norm(self.t_cog_CT)
        
        # Setting standard filter variables.
        fs = 120.0
        Gp = 0.1
        Gs = 3.0
        wp = 0.8
        ws = 1.1

        # load marker trace from file
        save_name = './Data/' + init_marker_ID + '_opti_marker_trace.npy'

        try:
           self.opti_marker_trace = np.load(save_name)
        
        # build marker trace from csv file
        except:
            marker_trace = marker_variable_id_linewise(test_path, init_marker_ID, test_metadata["type"], 40)
            inter_data = nan_helper(marker_trace)
            self.opti_marker_trace = plot_tiefpass(fs, Gp, Gs, wp, ws, inter_data)
            np.save(save_name, self.opti_marker_trace)
        
        # marker trace in different coordinate systems
        self.ct_marker_trace = [np.matmul(opti_pose,base_tracker.T_ct_def[:3,:3]) + base_tracker.T_ct_def[:3,3] for opti_pose in self.opti_marker_trace]

        # overwrite cog_traj_CT from tracker_bone
        self.cog_traj_CT = [self.ct_marker_trace \
                                + np.matmul(np.matmul(self.t_cog_CT, base_tracker.T_ct_def[:3,:3]), Quaternion(base_tracker.track_traj_opti[i,:4]).rotation_matrix) \
                                for i in range(len(self.ct_marker_trace))]
    
    def get_marker_pos_ct(self):
        '''Returns the relative marker positions in the CT coordinate system to the bone cog.'''
        with open(self.metadata["marker def CT"]) as jsonfile:
            data = json.load(jsonfile)
            # extract point infos
            point_data = data['markups'][0]['controlPoints']
        self.marker_pos_ct = [point['position'] for point in point_data]
        return self.marker_pos_ct

# %%
if __name__ == '__main__':

    #path = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'
    path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

    #raw_data = csv_test_load(path, '55')
    marker_ID = 'Unlabeled 2016'
    marker_ID = 'Unlabeled 2403'
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