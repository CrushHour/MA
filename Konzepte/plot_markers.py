from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import stl
import numpy as np
import sys
sys.path.append('./')
sys.path.append('./mujoco')
sys.path.append(r'C:/GitHub/MA/mujoco')
sys.path.append(r'C:/GitHub/MA')
import transformation_functions as tf

if __name__ == "__main__":

    test_metadata = tf.get_json('test_metadata.json')['Take 2023-01-31 06.11.42 PM.csv']
    hand_metadata = tf.get_json('hand_metadata.json')

    # %% Visualisierung der Marker und Tracker
    # calculate spheres
    # diese Berechnung gibt die Länge eines Fingers zurück.
    # es würde aber viel mehr sinn machen den Abstand zwischen den Joints
    # aus den mkr.json files zu nehmen.
    Tracker_ZF_DIP = tf.tracker_bone('ZF_DIP',test_path=test_metadata['path'])

    ZF_PIP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_PIP.stl")
    minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_PIP)
    d_ZF_DIP_PIP = np.linalg.norm([maxx-minx, maxy-miny, maxz-minz])

    ZF_MCP = stl.mesh.Mesh.from_file("./Data/STL/Segmentation_ZF_MCP.stl")
    minx, maxx, miny, maxy, minz, maxz = tf.stl_find_mins_maxs(ZF_MCP)
    d_ZF_MCP_PIP = np.linalg.norm([maxx-minx, maxy-miny, maxz-minz])
    ZF_Tracker_lst = []
    DAU_Tracker_lst = []
    name_lst = []

    radius_lst = []

    tf.plot_class(0,ZF_Tracker_lst,DAU_Tracker_lst,name_lst,radius_lst, save=False, show=True)

    interact(tf.plot_class, i = widgets.IntSlider(min=0,max=len(Tracker_ZF_DIP.track_traj_opt)-1,step=1,value=0),
            Trackers1 = widgets.fixed(ZF_Tracker_lst), 
            Trackers2 = widgets.fixed(DAU_Tracker_lst),
            names = widgets.fixed(name_lst),
            radius = widgets.fixed(radius_lst),
            show = widgets.fixed(True))
    #%%
    # test_points1 = [[2,-5,4], [5,6,7], [-10,0,3], [-3,11,13], [8,5,4]]
    # test_points2 = [[-1,4,3], [8,4,-3], [12,7,9], [4,-5,6], [-2,10,7]]