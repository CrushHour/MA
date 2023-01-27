#%% import
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
import csv
from mpl_toolkits.mplot3d import Axes3D

#%% transformation optitrack tracker to real tracker
path_csv = "/home/robotlab/Documents/GitHub/MA_Schote/MA/Data"

def get_opti_positions(filename):
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

    return name, opti_positions
	
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
    print('x:')
    print(x)
    for i in range(0,n):
            x2.append(opti_points[i][0])
            y2.append(opti_points[i][1])
            z2.append(opti_points[i][2])
    print('x2:')
    print(x2)
    ax.scatter(x2,y2,z2, c='r', marker='o')
    ax.plot([line_1[0][0],line_1[1][0]],[line_1[0][1],line_1[1][1]], zs=[line_1[0][2],line_1[1][2]], c='b')
    ax.plot([line_2[0][0],line_2[1][0]],[line_2[0][1],line_2[1][1]], zs=[line_2[0][2],line_2[1][2]], c='b')
    ax.plot([line_3[0][0],line_3[1][0]],[line_3[0][1],line_3[1][1]], zs=[line_3[0][2],line_3[1][2]], c='r')
    ax.plot([line_4[0][0],line_4[1][0]],[line_4[0][1],line_4[1][1]], zs=[line_4[0][2],line_4[1][2]], c='r')

    plt.show()


def get_min_max_dis(points):
    n = len(points)
    d_comp_max = 0
    d_comp_min = 100000000000000
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
    d_comp_max = 0
    d_comp_min = 100000000000000
    points_out = points
    pairs = []
    for i in range(n):
        print('i:', i)
        for j in range(n - 1, -1 + i, -1):
            if j != i:
                print('    j:', j)
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                dz = points[j][2] - points[i][2]

                d_diff = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                pairs.append([points[j], points[i], d_diff])

    pairs.sort(key=lambda x: x[2])

    return pairs

def compare_point_lists(pairs1, points1, pairs2, points2):
    distance_value_in_points1 = [[], [], [], [], []]
    distance_value_in_points2 = [[], [], [], [], []]

    # In der sortierten Liste bekommt jeder Punkt, einen Score für die Position an der er steht. Punkte die sich an
    # höreren Distanzen beteiligen erhalten mehr punkte. Dadurch erhält jeder Punkt einen eideutigen Score
    for i in range(len(pairs1)):
        for j in range(len(points1)):
            try:
                pairs1[i].index(points1[j])
                distance_value_in_points1[j].append(i + 1)
            except:
                pass
            try:
                pairs2[i].index(points2[j])
                distance_value_in_points2[j].append(i + 1)
            except:
                pass
    print(distance_value_in_points1)
    print(distance_value_in_points2)

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    # points_out = [x for _, x in sorted(zip(distance_value_in_points, points))]



class Tracker_3dicke:
    numTrackers = 5
    positions = [[0, 0, 75], [-42, 0, 46], [25, 0, 46], [0, 37, 41.5], [0, -44, 41.5]] # [[x,y,z],[x2,y2,z2],...]
    name, opti_positions = get_opti_positions('MakerJS_3dicke.csv')

class Tracker_Nico:
    numTrackers = 5
    positions = [[0, 0, 61], [-41, 0, 35], [20, 0, 35], [-10, 31, 35], [-10, -14, 35]] # [[x,y,z],[x2,y2,z2],...]
    name, opti_positions = get_opti_positions('Tracker Nico.csv')


#%% asd
if __name__ == '__main__':
    print(Tracker_Nico.opti_positions)
    v_max_tracker, v_min_tracker = get_min_max_dis(Tracker_Nico.positions)
    v_max_opti, v_min_opti = get_min_max_dis(Tracker_Nico.opti_positions)
    plot_ply(Tracker_Nico.positions, Tracker_Nico.opti_positions, v_max_tracker, v_min_tracker, v_max_opti, v_min_opti)
