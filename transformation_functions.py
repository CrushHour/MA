#%% import
import os
from datetime import datetime
import Konzepte.trackers as trackers
import pandas as pd
import numpy as np
import stl
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ipywidgets import interact, fixed
import ipywidgets as widgets
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
import csv
import json 
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import distance
from scipy.signal import butter, buttord, filtfilt
from tqdm import tqdm

#%% transformation opttrack tracker to real tracker
path_csv = "Data"
ct_folder = "Slicer3D"

def plot_analogs(path):
    data = get_json(path)
    sensor_data = data['observation']['analogs']
    time = data['time']
    labels = ['Thumb Spreader', 'Thumb flexor', 'Flexor 2 index finger', 'Flexor 1 pointer finger', 'Extensor 1 index finger', 'Extensor 2 thumb', 'Extensor 2 index finger', 'Extensor 1 thumb','Temperature']
    thumb = [0,1,5,7]
    index = [2,3,4,6]
    pos_x = np.arange(max(time), step=5000) # type: ignore
    x = [int(i/1000) for i in pos_x]

    for i in thumb:
        plt.plot(time,sensor_data[i]['force'], label=labels[i])
    plt.xticks(pos_x,x)
    plt.xlabel('time [sec]')
    plt.ylabel('force [N]') # ist die Kraft woirklich in N?
    plt.legend()
    plt.title('Thumb')
    plt.grid()
    plt.show()
    plt.close()

    for i in index:
        plt.plot(time,sensor_data[i]['force'], label=labels[i])
    plt.xticks(pos_x,x)
    plt.xlabel('time [sec]')
    plt.ylabel('force [N]') # ist die Kraft woirklich in N?
    plt.legend()
    plt.title('Index Finger')
    plt.grid()
    plt.show()
    plt.close()

def plot_analogs_angles(angles=[], flexor=[], extensor=[], time=[], step_size=5000, start = 0, end = 0, legend=[], title='', save_plots=False):
    time = time[start:end]
    angles = [angle[start:end] for angle in angles]
    flexor = [flex[start:end] for flex in flexor]
    extensor = [ext[start:end] for ext in extensor]

    pos_x = np.arange(max(time), step=step_size)  # type: ignore
    start_ticks = int(time[0]/step_size)
    pos_x = pos_x[start_ticks:]
    x = [int(i / 1000) for i in pos_x]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(time, np.transpose(angles))
    ax2.plot(time, np.transpose(flexor))
    ax3.plot(time, np.transpose(extensor))
    ax1.set_ylabel('angle [°]')
    ax2.set_ylabel('force [N]')
    ax3.set_ylabel('force [N]')
    ax3.set_xlabel('time [sec]')
    ax1.set_title(title)
    ax2.set_title('flexor')
    ax3.set_title('extensor')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    plt.xticks(pos_x, x)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    plt.close()



def get_opt_positions(filename):
    '''Loads tracker information of a given tracker export file.'''
    path = path_csv + "/" + filename
    opt_positions = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == 'Name':
                name = row[1]
            if row[0] == 'Point':
                opt_positions.append([])
                for i in range(2,5):
                    opt_positions[-1].append(float(row[i]))

    #return name, opt_positions
    return opt_positions

def csv_test_load(testrun_path, tracker_designation_motive):
    '''This function is suppose to read in the trakcing data of a single tracker
    of a specified testrun from an motive .csv export.'''
    df = pd.read_csv(testrun_path, header=2, low_memory=False)
    start_coloum = df.columns.get_loc(tracker_designation_motive)
    data = df.values[3:,start_coloum:start_coloum+8]
    data = np.array([list(map(float, i)) for i in data])
    return data

def json_test_load(path='./Data/optitrack-20230130-234800.json', initialID=''):
    '''This function is suppose to read in the trakcing data of a single tracker'''
    
    f = open(path)
    data = json.load(f)
    df = pd.DataFrame(data)
    f.close()
    
    header = []
    len_header = 0
    
    for i in range(df.shape[1]):
        try:
            header.append(df[i][0]['name'])
            header.append(df[i][0]['name']+'.1')
            header.append(df[i][0]['name']+'.2')
            header.append(df[i][0]['name']+'.3')
            header.append(df[i][0]['name']+'.4')
            header.append(df[i][0]['name']+'.5')
            header.append(df[i][0]['name']+'.6')
            len_header += 1
        except:
            continue

    xyz = np.zeros((df.shape[0],len(header)))
    
    for i in range(df.shape[0]):
        for j in range(len_header):
            try:
                xyz[i,j*7] = df[j][i]['qx']
                xyz[i,j*7+1] = df[j][i]['qy']
                xyz[i,j*7+2] = df[j][i]['qz']
                xyz[i,j*7+3] = df[j][i]['qw']
                xyz[i,j*7+4] = df[j][i]['x']*1000
                xyz[i,j*7+5] = df[j][i]['y']*1000
                xyz[i,j*7+6] = df[j][i]['z']*1000

            except:
                xyz[i,j*7] = np.nan
                xyz[i,j*7+1] = np.nan
                xyz[i,j*7+2] = np.nan
                xyz[i,j*7+3] = np.nan
                xyz[i,j*7+4] = np.nan
                xyz[i,j*7+5] = np.nan
                xyz[i,j*7+6] = np.nan
    
    df = pd.DataFrame(xyz, columns=header)

    if initialID != '':
        next_col = df.columns.get_loc(initialID)
        df = df.iloc[:,next_col:next_col+7]

    return df

def phoenix_test_load(path='Data/test_01_31/2023_01_31_18_12_48.json', initialID=''):
    '''This function is suppose to read in the trakcing data of a single tracker'''
    
    f = open(path)
    data = json.load(f)
    time = data['time']
    df = pd.DataFrame(data['observation']['rigid_bodies'])
    f.close()

    body_ids = ['1028', '1031', '1032', '1029', '1030', '1027']
    
    header = []
    len_header = 0
    
    for i in range(df.shape[0]):
        try:
            header.append(body_ids[i])
            header.append(body_ids[i]+'.1')
            header.append(body_ids[i]+'.2')
            header.append(body_ids[i]+'.3')
            header.append(body_ids[i]+'.4')
            header.append(body_ids[i]+'.5')
            header.append(body_ids[i]+'.6')
            len_header += 1
        except:
            continue

    xyz = np.zeros((len(time),len(header)))
    
    for j in range(len_header):
        xyz[:,j*7] = df['qx'][j]
        xyz[:,j*7+1] = df['qy'][j]
        xyz[:,j*7+2] = df['qz'][j]
        xyz[:,j*7+3] = df['qw'][j]
        xyz[:,j*7+4] = [i*1000 for i in df['x'][j]]
        xyz[:,j*7+5] = [i*1000 for i in df['y'][j]]
        xyz[:,j*7+6] = [i*1000 for i in df['z'][j]]

    df = pd.DataFrame(xyz, columns=header)

    if initialID != '':
        next_col = df.columns.get_loc(initialID)
        df = df.iloc[:,next_col:next_col+7]

    return df, time

def marker_variable_id_linewise(testrun_path, initialID=None, dtype="csv", d_max = 560):
    
    initialID = str(initialID)
    df = pd.DataFrame()
    
    # 1. load data
    if dtype == "json":
        print("rebuilding json to be formated as csv")
        df = json_test_load(testrun_path, initialID)

    else:
        df = pd.read_csv(testrun_path, header=2, low_memory=False)
    
    next_col = df.columns.get_loc(initialID)

    # 2. variablen initiieren. Position values start at line 4 (counting from 1).
    start_line = 3
    dis_list = []
    start_value = df.values[start_line,next_col:next_col+3]

    """If condition to be able to catch trackers, that a not visiable immidiatly"""
    if np.isnan(float(start_value[0])):
        # find cells that are not empty, so you know where to start
        filled_cells = np.where(pd.notna(df.iloc[:,next_col]))
        start_line = int(filled_cells[0][3])
        start_value = df.values[start_line,next_col:next_col+3]
        print("start value:", start_value)
        print("start line:", start_line)

    #added_data = np.zeros((df.shape[0]-start_line,3)) 
    added_data = np.zeros((df.shape[0]-3,3)) 
    added_data[0,:] = start_value
    added_data[start_line-3,:] = start_value
    last_signal = added_data[0,:]

    # 3. Start Zeilenschleife
    #for k in tqdm(range(1,added_data.shape[0])):
    for k in tqdm(range(start_line-3,added_data.shape[0])):
        
        min_dis = np.inf
        current_dis = np.inf

        #4.1 Zeilen die zu vergleichen sind, werden ausgewählt und in ein Array geschrieben. Dieses Array wird dann in die nächste Schleife gegeben. Es heißt value.
        line = df.values[k+3,:]
        #line = df.values[k,:]
        line = np.array(list(map(float, line)))

        # Spalten. If there is no value in any of the coloums of the line, np.nan will be added.
        values_to_add = [np.nan, np.nan, np.nan]

        for j in range(next_col,len(line),3):
            value = line[j:j+3]
            
            if np.isnan(value[0]) or np.isnan(value[1]) or np.isnan(value[2]):
                continue         

            else:
                current_dis = np.absolute(np.linalg.norm(last_signal) - np.linalg.norm(value))
                if current_dis < min_dis:
                    min_dis = current_dis.copy()
                    values_to_add = value.copy()
        
        # falls die Minimale Distanz zum nächsten Punkt den Grenzwert überschreitet,
        # wird der Punkt nicht in die Ausgabeliste eingetragen.
        added_data[k,:] = values_to_add

    #print(dis_list)
    print("len added_data for marker %s:" % initialID, len(added_data))
    return added_data

def marker_variable_id_linewise_march28(testrun_path, initialID=None, dtype="csv"):
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
            last_signal == values_to_add #type: ignore
        
        # falls die Minimale Distanz zum nächsten Punkt den Grenzwert überschreitet,
        # wird der Punkt nicht in die Ausgabeliste eingetragen.
        dis_list.append(min_dis)
        if min_dis >= 56:
            added_data[k,:] = [np.nan, np.nan, np.nan]
        else:
            added_data[k,:] = values_to_add

    #print(dis_list)
    return added_data

def perpendicular_vector(v):
    '''on possible solution for 3x1 vectors.'''
    vp = np.array([0, v[2], -v[1]])
    return vp

def plot_angles(angles, time, step_size, legend, title, save_plots=False):

    pos_x = np.arange(max(time), step=step_size) # type: ignore
    x = [int(i) for i in pos_x]
    if max(pos_x) > 1000.0:
        x = [int(i/1000) for i in pos_x]

    for i in range(len(angles)):
        #default_x = np.arange(len(angles[i]))
        #plt.plot(default_x, angles[i])
        plt.plot(time, angles[i])

    plt.xticks(pos_x,x)
    plt.legend(legend, loc='upper right')
    plt.ylabel('angle [°]')
    plt.xlabel('time [sec]')
    plt.title(title)
    plt.grid(True)
    if save_plots:
        plt.savefig('./plots/angles/' + title + '.svg', dpi=1200)
    else:
        plt.show()
    plt.close()

def plot_quaternion(quaternion, title, save_plots=False):
    pos_x = np.arange(len(quaternion), step=600) # type: ignore
    x = [int(pos_x[i]/120) for i in range(len(pos_x))]

    q_lst = np.zeros((len(quaternion),4))
    default_x = np.arange(len(quaternion))

    for i in range(len(quaternion)):
        q_lst[i] = quaternion[i].elements
        q_lst = np.array(q_lst)
    
    plt.plot(default_x, q_lst[:,0], label="s")
    plt.plot(default_x, q_lst[:,1], label="x")
    plt.plot(default_x, q_lst[:,2], label="y")
    plt.plot(default_x, q_lst[:,3], label="z")

    plt.xticks(pos_x,x)
    plt.legend(loc='upper right')
    plt.xlabel('time [sec]')
    plt.title(title)
    if save_plots:
        plt.savefig('./plots/angles/' + title + '.svg', dpi=1200)
    else:
        plt.show()
    plt.close()

def plot_ply(tracker_points, opt_points, line_1, line_2, line_3, line_4):
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
            x2.append(opt_points[i][0])
            y2.append(opt_points[i][1])
            z2.append(opt_points[i][2])
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
    customcolors = longcolors + morecolors + ['darkblue', 'darkgreen', 'darkred', 'darkcyan', 'darkmagenta', 'darkyellow', 'darkgray', 'darkolive', 'darkcyan']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    # Plot known tracker points
    j = len(Trackers1)
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
    minx = maxx = miny = maxy = minz = maxz = float('nan')
    for i, p in enumerate(obj.points):
        # p contains (x, y, z)
        if i== 0:
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
    v_max = []
    v_min = []

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

    '''An dieser Stelle soll die Liste der Punkte aus dem CT (2) anhand der Punkte aus dem opt-Export (1)
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

def plot_tiefpass(marker_data, title: str='', fs: float =120, Gp: float = 0.1, Gs: float=3.0, wp: float=0.8, ws: float=1.1):
    order, wn = buttord(wp, ws, Gp, Gs)
    y = butter_lowpass_filter(marker_data, wn, fs, order)
    
    # Plotting
    fig = plt.subplot(1, 1, 1)
    fig.plot(marker_data, color = 'lightgrey', linewidth=0.25, label='marker data')

    fig.plot(y[:,0], 'r-', linewidth=0.5, label='filtered data x')
    fig.plot(y[:,1], 'g-', linewidth=0.5, label='filtered data y')
    fig.plot(y[:,2], 'b-', linewidth=0.5, label='filtered data z')

    plt.xlabel('Time [120 Hz]')
    plt.title(title)
    fig.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.subplots_adjust(hspace=0.35)
    now = datetime.now()
    plot_file_title = "marker_" + now.strftime("%d_%m_%Y_%H_%M_%S")
    # plt.savefig(plot_file_title + ".pdf", format="pdf")
    plt.show()
    plt.close()
    return y

def hist_filter(T_in, n_std = 3):
    shape = T_in.shape
    
    if len(shape) == 3:
        y = T_in[:,0,shape[-1]]
    else:
        y = T_in[:,shape[-1]]
    
    dy = np.gradient(y)
    
    indicies = np.where(dy > np.mean(dy)+n_std*np.std(dy))
    indicies2 = np.where(dy < np.mean(dy)-n_std*np.std(dy))

    x_new = np.copy(y)
    for i in indicies:
        x_new[i] = y[i-1]
    for i in indicies2:
        x_new[i] = y[i-1]
    T_in[:,shape[-1]] = x_new
    return T_in

# %%
def angle_axis(v1,v2,axis):

    # Normalize the vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    axis_normalized = axis / np.linalg.norm(axis)

    # Project the vectors onto a plane perpendicular to the axis
    v1_projected = v1_normalized - np.dot(v1_normalized, axis_normalized) * axis_normalized
    v2_projected = v2_normalized - np.dot(v2_normalized, axis_normalized) * axis_normalized

    # Calculate the dot product of the projected vectors
    dot_product_projected = np.dot(v1_projected, v2_projected)

    # Calculate the angle between the vectors
    angle = np.arccos(dot_product_projected) * (180 / np.pi)

    print("The angle between the vectors is:", angle)
    return angle

def angle_projectet(v1,v2,normal):
    '''Normalisiere den Normalenvektor der Ebene, um sicherzustellen, dass er eine Länge von 1 hat.

    Bestimme den Richtungsvektor des gegebenen Vektors im Raum. Du kannst dies tun, indem du den gegebenen Vektor vom Ursprung aus subtrahierst.

    Berechne das Skalarprodukt zwischen dem Richtungsvektor des gegebenen Vektors und dem normalisierten Normalenvektor der Ebene.

    Multipliziere das Skalarprodukt aus Schritt 3 mit dem normalisierten Normalenvektor der Ebene.

    Subtrahiere das Ergebnis aus Schritt 4 vom gegebenen Vektor, um den Schattenpunkt zu erhalten.'''

    # Normalisiere den Normalenvektor der Ebene, um sicherzustellen, dass er eine Länge von 1 hat.
    normal = normal / np.linalg.norm(normal)

    # Bestimme den Richtungsvektor des gegebenen Vektors im Raum. Du kannst dies tun, indem du den gegebenen Vektor vom Ursprung aus subtrahierst.
    #v1 = v1 - np.array([0,0,0])
    #v2 = v2 - np.array([0,0,0])

    # Berechne das Skalarprodukt zwischen dem Richtungsvektor des gegebenen Vektors und dem normalisierten Normalenvektor der Ebene.
    dot_product = [np.dot(v1, normal), np.dot(v2, normal)]

    # Multipliziere das Skalarprodukt aus Schritt 3 mit dem normalisierten Normalenvektor der Ebene.
    v1_projected = dot_product[0] * normal
    v2_projected = dot_product[1] * normal

    # Subtrahiere das Ergebnis aus Schritt 4 vom gegebenen Vektor, um den Schattenpunkt zu erhalten.
    v1_shadow = v1 - v1_projected
    v2_shadow = v2 - v2_projected

    angle = angle_between(v1_shadow, v2_shadow)

    return angle

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

def nan_helper_1d(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_1d(y):
    nans, x= nan_helper_1d(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

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
        file_path = path + key[:3] + '_A' + key[3:] + '.mrk.json'        # 1. load mounting points
        with open(file_path) as jsonfile:
            data = json.load(jsonfile)

        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        helper_points[key] = [point['position'] for point in point_data]
        helper_points[key].append(np.mean(helper_points[key], axis=0))
    return helper_points[finger_name]

def get_single_joint_file(file_path = ['./Slicer3D/DAU_COG.mrk.json']):
    '''Returns the joint points for the finger_name.'''
    if file_path == '':
        return np.array([np.nan,np.nan,np.nan])
    else:
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
            #joint_pos.append(np.array(helper_points))
    return joint_pos

def get_test_metadata(name):
    '''Returns the metadata of the test.'''
    try:
        with open('test_metadata.json') as json_data:
            d = json.load(json_data)
            metadata = d[name]
            json_data.close()
    except:
        with open(r'C:/GitHub/MA/test_metadata.json') as json_data:
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
class tracker_bone():
    
    def __init__(self, finger_name = "", test_path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv') -> None:
        self.finger_name = finger_name
        print("Finger:", self.finger_name)
        self.metadata = self.get_metadata()
        self.finger_name = finger_name
        self.defname = self.metadata['tracker def opt']
        self.ctname = self.metadata['tracker def CT']
        
        # read definition file into memory
        self.read_markerdata()

        # read jsonfile from ct into memory
        self.read_ctdata()
        # catch the case where the ct file is about a marker instead of a tracker
        
        # get the information of the stl file
        stl_data = stl.mesh.Mesh.from_file(self.metadata['stl'])
        self.volume, self.cog_stl, self.inertia = stl_data.get_mass_properties()
        
        '''This Part is only for full trackers.'''
        if self.metadata["tracking"] == "Tracker":
            self.marker_pos_def, self.marker_pos_ct = sort_points_relative(self.marker_pos_def, self.marker_pos_ct)
            self.T_ct_def = self.calculate_transformation_matrix(np.array(self.marker_pos_def), np.array(self.marker_pos_ct)) # P nach Q
            self.T_def_ct = self.invert_T(self.T_ct_def)
            self.t_ct_tr = self.T_ct_def
            # Get the trajectory of the tracker from the test data
            if test_path.endswith('00.json') and test_path.find('optitrack'):
                self.track_traj_opt = json_test_load(test_path, self.metadata['tracker ID'])
                self.track_traj_opt = self.track_traj_opt.values
                self.time = np.arange(0, len(self.track_traj_opt)/120, 1/120)
            elif test_path.endswith('.json'):
                self.track_traj_opt, self.time = phoenix_test_load(test_path, self.metadata['tracker ID'])
                self.track_traj_opt = self.track_traj_opt.values
            else:
                self.track_traj_opt = csv_test_load(test_path, self.metadata['tracker name'])
                self.time = np.arange(0, len(self.track_traj_opt)/120, 1/120)

            self.track_traj_opt = nan_helper(self.track_traj_opt)
            #self.track_traj_opt = self.replace_outliers(self.track_traj_opt)
            #self.track_traj_opt = plot_tiefpass(inter_data, self.finger_name, wp = 0.8, ws = 1.1)
            self.track_traj_opt = self.delete_outliers(self.track_traj_opt, 2.5)
            self.track_traj_opt = nan_helper(self.track_traj_opt)

            
            # initialize the transformation matrix
            self.T_opt_i = np.zeros((len(self.track_traj_opt),4,4))
            self.T_opt_ct = np.zeros((len(self.track_traj_opt),4,4))
            
            # T from timestamp i to opt coordinate system
            for i in range(len(self.track_traj_opt)):
                # T from timestamp i to opt coordinate system
                s = self.track_traj_opt[i,3]
                v = self.track_traj_opt[i,:3]
                q = Quaternion(scalar=s, vector=v)
                t = self.track_traj_opt[i,4:7]
                self.T_opt_i[i] = np.eye(4)
                self.T_opt_i[i,:3,:3] = q.rotation_matrix
                self.T_opt_i[i,:3,3] = t
                #self.T_opt_i[i,3,3] = 1

                # T from CT coordinate system to timestamp i
                self.T_opt_ct[i,:,:] = self.T_opt_i[i,:,:] @ self.T_def_ct

            self.helper_points = []
            for joint in self.metadata['joints']:
                self.helper_points.append(get_single_joint_file(joint))

            if not np.isnan(self.helper_points[1]).any():
                self.t_proxi_aussen_CT = self.helper_points[1][1]
                self.T_proxi_aussen_CT = np.eye(4)
                self.T_proxi_aussen_CT[:3,3] = self.t_proxi_aussen_CT
                self.T_proxi_aussen_opt = np.zeros((len(self.track_traj_opt),4,4))

                self.t_proxi_innen_CT = self.helper_points[1][0]
                self.T_proxi_innen_CT = np.eye(4)
                self.T_proxi_innen_CT[:3,3] = self.t_proxi_innen_CT
                self.T_proxi_innen_opt = np.zeros((len(self.track_traj_opt),4,4))

                self.t_proxi_CT = np.mean(self.helper_points[1], axis=0)
                self.T_proxi_CT = np.eye(4)
                self.T_proxi_CT[:3,3] = self.t_proxi_CT
                self.T_proxi_opt = np.zeros((len(self.track_traj_opt),4,4))

                for i in range(len(self.track_traj_opt)):
                    self.T_proxi_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_proxi_CT
                    self.T_proxi_aussen_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_proxi_aussen_CT
                    self.T_proxi_innen_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_proxi_innen_CT
            else:
                print('No proximal joint found.')

            if not np.isnan(self.helper_points[0]).any():
                self.t_dist_aussen_CT = self.helper_points[0][1]
                self.T_dist_aussen_CT = np.eye(4)
                self.T_dist_aussen_CT[:3,3] = self.t_dist_aussen_CT
                self.T_dist_aussen_opt = np.zeros((len(self.track_traj_opt),4,4))

                self.t_dist_innen_CT = self.helper_points[0][0]
                self.T_dist_innen_CT = np.eye(4)
                self.T_dist_innen_CT[:3,3] = self.t_dist_innen_CT
                self.T_dist_innen_opt = np.zeros((len(self.track_traj_opt),4,4))

                self.t_dist_CT = np.mean(self.helper_points[0],axis=0)
                self.T_dist_CT = np.eye(4)
                self.T_dist_CT[:3,3] = self.t_dist_CT
                self.T_dist_opt = np.zeros((len(self.track_traj_opt),4,4))
                
                for i in range(len(self.track_traj_opt)):
                    self.T_dist_aussen_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_dist_aussen_CT
                    self.T_dist_innen_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_dist_innen_CT
                    self.T_dist_opt[i,:,:] = self.T_opt_ct[i,:,:] @ self.T_dist_CT
                    

            else:
                print('No distal joint found.')

            self.v_opt = np.subtract(self.T_dist_opt[:,:3,3],self.T_proxi_opt[:,:3,3])


    def get_metadata(self):
        '''Returns the metadata of the Phalanx.'''
        try:
            with open('hand_metadata.json') as json_data:
                d = json.load(json_data)
                metadata = d[self.finger_name]
                json_data.close()
            return metadata
        except:
            with open(r'C:/GitHub/MA/hand_metadata.json') as json_data:
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
    
    def calculate_transformation_matrix(self, markers1=None, markers2=None):
        """uses Kabsch algorithm to calculate transformation matrix between system 1 and 2"""

        # Convert lists of markers to arrays
        markers1 = np.array(markers1)
        markers2 = np.array(markers2)

        # Center the markers at the origin
        markers1_mean = np.mean(markers1, axis=0)
        markers2_mean = np.mean(markers2, axis=0)
        markers1 -= markers1_mean
        markers2 -= markers2_mean

        # Calculate the cross-covariance matrix (H = P^T * Q)
        H = np.dot(markers1.T, markers2)

        # Calculate the singular value decomposition
        U, S, V_T = np.linalg.svd(H)

        # decide whether we need to correct our rotation matrix to ensure a right-handed coordinate system
        # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        d = np.sign(np.linalg.det(np.dot(V_T.T, U.T)))

        D = np.eye(3)
        D[2, 2] = d

        # Calculate the rotation matrix
        R = V_T.T @ D @ U.T

        # Check for reflection
        if np.linalg.det(R) < 0:
            V_T[2, :] *= -1
            R = V_T.T @ U.T # Julian

        # Calculate the translation vector
        t = markers2_mean - np.dot(markers1_mean, R.T)

        # Concatenate the rotation and translation matrices
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t
        """
        This matrix points FROM def TO ct
        I checked and can cofirm.
        """
        return transformation_matrix
    
    def kabsch(self, markers1=None, markers2=None):

        P = np.array(markers1)
        Q = np.array(markers2)

        #center points
        P_mean = np.mean(P, axis=0)
        Q_mean = np.mean(Q, axis=0)
        P -= np.mean(P, axis=0)
        Q -= np.mean(Q, axis=0)
        

        # Computation of the covariance matrix
        H = np.dot(np.transpose(P), Q)
        
        # Singular value decomposition
        U, S, V_t = np.linalg.svd(H)

        # Calculation of R
        d = np.sign(np.linalg.det(np.dot(V_t.T, U.T)))
        D = np.eye(3)
        D[2, 2] = d
        R = np.dot(V_t.T, np.dot(D, U.T))

        # Calculation of t
        t = Q_mean - np.dot(R, P_mean)

        # build transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        return transformation_matrix

    
    def replace_outliers(self, data, n_std = 5):
        dis = np.array([])
        for i in range(len(data)-1):
            dis = np.append(dis, np.linalg.norm(np.subtract(data[i],data[i+1])))
        threshold = np.mean(dis)+n_std*np.std(dis, axis=0)
        clean_data = []
        last_valid = data[0]
        dis_lst = []

        for i in range(len(data)):
            dis = np.linalg.norm(data[i] - last_valid)
            dis_lst.append(dis)
            if dis <= threshold:
                clean_data.append(data[i])
                last_valid = data[i]
            else:
                clean_data.append(last_valid)

        return np.array(clean_data)
    
    def delete_outliers(self,data, n_std = 2.0):
        data = np.array(data)
        no_value = np.where(np.isnan(data)==False)
        if len(no_value) >= 1:
            ValueError('NaN values in data.')
        indexes = np.where(abs(data - np.mean(data)) > n_std * np.std(data))

        if indexes[0].size > 0:
            for index in indexes[0]:
                try:
                    data[index] = [np.nan]*len(data[index])
                except:
                    data[index] = np.nan
    
        return data

    def delete_outliers_local(self,data, n_std = 2.0, window = 1500):
        data = np.array(data)
        no_value = np.where(np.isnan(data)==False)
        if len(no_value) >= 1:
            ValueError('NaN values in data.')
        indexes = []
        iterations = int(len(data)/window)
        for i in range(iterations-1):
            indexes = np.where(abs(data[i*window:(i+1)*window] - np.mean(data[i*window:(i+1)*window])) > n_std * np.std(data[i*window:(i+1)*window]))

        
            if indexes[0].size > 0:
                for index in indexes[0]:
                    try:
                        data[i*window+index] = [np.nan]*len(data[index])
                    except:
                        data[i*window+index] = np.nan
    
        return data
    
class marker_bone():
    '''Class for the bones with markers on top. Inherits from tracker_bone.'''
    def __init__(self, finger_name = "", test_path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv', init_marker_ID = "Unlabeled ...") -> None:
        
        #base_tracker = tracker_bone("ZF_midhand", test_path=test_path)
        self.finger_name = finger_name
        print("Marker on", self.finger_name)
        self.testname = os.path.split(test_path)[1]
        test_metadata = get_test_metadata(self.testname)
        self.metadata = self.get_metadata()

        

        self.joints = []
        for i in range(len(self.metadata['joints'])):
            self.joints.append(get_single_joint_file(self.metadata['joints'][i]))

        self.T_proxi_CT = np.array([np.eye(4),np.eye(4)])
        self.T_proxi_CT[0,:3,3] = self.joints[1][0]
        self.T_proxi_CT[1,:3,3] = self.joints[1][1]

        self.T_dist_CT = np.array([np.eye(4),np.eye(4)])
        self.T_dist_CT[0,:3,3] = self.joints[0][0]
        self.T_dist_CT[1,:3,3] = self.joints[0][1]

        # get stl information
        stl_data = stl.mesh.Mesh.from_file(self.metadata['stl'])
        self.volume, self.cog_stl, self.inertia = stl_data.get_mass_properties()
        
        # Setting standard filter variables.
        fs = 120.0
        Gp = 0.1
        Gs = 3.0
        wp = 0.8
        ws = 1.1

        # load marker trace from file
        save_name = './Data/' + init_marker_ID + '_opt_marker_trace.npy'
        
        # get marker information
        if self.metadata["marker def CT"] != "" and init_marker_ID != "Unlabeled ..." and init_marker_ID != "" and init_marker_ID != " ":
            self.marker_pos_ct = self.get_marker_pos_ct()

            try:
                self.opt_marker_trace = np.load(save_name)
            
            # build marker trace from csv file
            except:
                marker_trace = marker_variable_id_linewise_march28(test_path, init_marker_ID, test_metadata["type"])
                inter_data = nan_helper(marker_trace)
                self.opt_marker_trace = plot_tiefpass(inter_data, init_marker_ID, fs, Gp, Gs, wp, ws)
                np.save(save_name, self.opt_marker_trace)
        else:
            self.opt_marker_trace = np.zeros((int(test_metadata["length"]),3))

        #prepare matrices for transformation
        self.T_proxi_opt = np.zeros((int(test_metadata["length"]),2,4,4))
        self.T_dist_opt = np.zeros((int(test_metadata["length"]),2,4,4))

        self.T_opt_ct = np.zeros((int(test_metadata["length"]),4,4))
        self.v_opt = np.zeros((int(test_metadata["length"]),3))
    
    def replace_outliers(self, data, threshold = 0, compare_steps = 20, n_std = 5):
        dis = np.array([])
        for i in range(len(data)-1):
            dis = np.append(dis, np.linalg.norm(np.subtract(data[i],data[i+1])))
        if threshold == 0:
            threshold = np.mean(dis)+n_std*np.std(dis, axis=0)
        print('treshold =', threshold)
        clean_data = []
        last_valid = data[0]
        dis_lst = []

        clean_data.append(data[:compare_steps])

        for i in range(compare_steps,len(data)):
            dis = np.linalg.norm(data[i] - last_valid)
            dis_lst.append(dis)
            if dis <= threshold:
                clean_data.append(data[i])
                last_valid = data[i-compare_steps]
            else:
                #clean_data.append(last_valid)
                clean_data.append([np.nan,np.nan,np.nan])

        print(max(dis_lst))
        return np.array(clean_data)

    
    def get_marker_pos_ct(self):
        '''Returns the relative marker positions in the CT coordinate system to the bone cog.'''
        with open(self.metadata["marker def CT"]) as jsonfile:
            data = json.load(jsonfile)
            # extract point infos
            point_data = data['markups'][0]['controlPoints']
        self.marker_pos_ct = [point['position'] for point in point_data]
        return self.marker_pos_ct
    
    def get_metadata(self):
        '''Returns the metadata of the Phalanx.'''
        with open('hand_metadata.json') as json_data:
            d = json.load(json_data)
            metadata = d[self.finger_name]
            json_data.close()
        return metadata
    
    def update_joints(self, t):
        self.T_proxi_opt[t,0] = self.T_opt_ct[t] @ self.T_proxi_CT[0]
        self.T_proxi_opt[t,1] = self.T_opt_ct[t] @ self.T_proxi_CT[1]
        self.T_dist_opt[t,0] = self.T_opt_ct[t] @ self.T_dist_CT[0]
        self.T_dist_opt[t,1] = self.T_opt_ct[t] @ self.T_dist_CT[1]

# %%
if __name__ == '__main__':

    #path = r'C:\\GitHub\\MA\\Data\test_01_31\\Take 2023-01-31 06.11.42 PM.csv'
    path = './Data/test_01_31/Take 2023-01-31 06.11.42 PM.csv'

    #raw_data = csv_test_load(path, '55')
    marker_ID = 'Unlabeled 2016'
    #marker_ID = 'Unlabeled 2403'
    marker_data = marker_variable_id_linewise(path, marker_ID, "csv", 40)

    inter_data = nan_helper(marker_data)
    # Setting standard filter variables.
    fs = 120.0
    Gp = 0.1
    Gs = 3.0
    wp = 0.8
    ws = 1.1

    pic = plot_tiefpass(inter_data, 'Test', fs, Gp, Gs, wp, ws, )
    filtered_data = interact(plot_tiefpass,marker_data = fixed(inter_data), title = marker_ID,
                                    fs = fixed(fs),
                                    Gp = widgets.FloatSlider(value=Gp, min=0,max=2,step=0.1),
                                    Gs = widgets.FloatSlider(value=Gs, min=0,max=120,step=1), 
                                    wp = widgets.FloatSlider(value=wp, min=0,max=2,step=0.05), 
                                    ws = widgets.FloatSlider(value=ws, min=0,max=2,step=0.05)
                                        )
    test_metadata = get_test_metadata('Take 2023-01-31 06.11.42 PM.csv')
    Tracker_ZF_DIP = tracker_bone('ZF_DIP',test_path=test_metadata['path'])
    markers55 = [[116.838463, -106.912125, -5.724374],[111.952942, -142.248764, -17.220221],[121.998627, -124.245445, 11.670587],[148.879791, -143.25061, -2.70425],[143.807617, -113.712471, 0.872637]] # [x,y,z], Zeitpunkt 0
    p_mess, p_def = trackers.return_sorted_points(markers55, Tracker_ZF_DIP.marker_pos_def)
    s = Tracker_ZF_DIP.track_traj_opt[0,3]
    v = Tracker_ZF_DIP.track_traj_opt[0,:3]
    q = Quaternion(scalar=s, vector=v)
    q1 = q.inverse
    t = Tracker_ZF_DIP.track_traj_opt[0,4:7]
    T_i_k = np.eye(4)
    T_i_k[:3,:3] = q.rotation_matrix
    T_i_k[:3,3] = t
    T_k_i = Tracker_ZF_DIP.invert_T(T=T_i_k)
    T_i_markers55 = np.zeros((5,4,4))
    out = []
    for marker in markers55:
        T_i_marker = np.eye(4)
        T_i_marker[:3,3] = marker
        inter = T_k_i @ T_i_marker
        out.append(inter[:3,3])
        print(out[-1])
    print('-------------')
    print([str(pos) for pos in Tracker_ZF_DIP.marker_pos_def])
    print('-------------')
    print(np.mean(out, axis=0))
    '''Test bestanden :)'''

    #   Tracker_3dicke:
    #       numTrackers = 5
    #       positions = [[0, 0, 75], [-42, 0, 46], [25, 0, 46], [0, 37, 41.5], [0, -44, 41.5]] # [[x,y,z],[x2,y2,z2],...]
    #       name, opt_positions = get_opt_positions('MakerJS_3dicke.csv')
    #
    #    Tracker_Nico:
    #        numTrackers = 5
    #        positions = [[0, 0, 61], [-41, 0, 35], [20, 0, 35], [-10, 31, 35], [-10, -14, 35]] # [[x,y,z],[x2,y2,z2],...]
    #        name, opt_positions = get_opt_positions('Tracker Nico.csv')
    sort_points_relative([[1,1,1],[4,4,4],[10,10,10]],[[11,11,11],[7,7,7],[2,2,2]])