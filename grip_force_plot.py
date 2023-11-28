# %% import and functions

import numpy as np
import json
import sklearn.cluster as skc
import transformation_functions as tf
import matplotlib.pyplot as plt

def cluster_force_graph(x,y):
    plt.scatter(x,y)
    plt.show()
    return None

def find_timestamp_to_cluster_center(center, cluster_data):
    """
    Finds the closest point in data_points to target_point.

    :param data_points: A list of tuples, where each tuple represents a 7-dimensional point.
    :param target_point: A tuple representing the target 7-dimensional point.
    :return: The point in data_points closest to target_point.
    """
    def euclidean_distance(point1, point2):
        return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    closest_point = None
    min_distance = float('inf')
    itemindex = np.inf

    for point in cluster_data:
        distance = euclidean_distance(point, center)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
            itemindex = np.where(cluster_data == closest_point)
    return itemindex, closest_point

# %% load results
path = r'C:\GitHub\MA\Data\results\output_2023_01_31_18_12_48.json'
data = tf.get_json(path)
ft_norm = np.array(data['ft_norm'])
sum_tendon_force = np.array(data['sum_tendon_force'])
alpha = np.array(data['alpha'])
beta = np.array(data['beta'])
gamma = np.array(data['gamma'])
delta = np.array(data['delta'])
epsilon = np.array(data['epsilon'])
iota = np.array(data['iota'])
theta = np.array(data['theta'])

# np.where für F > 0,3 N trehshold
force_indexes = np.where(ft_norm > 0.3)[0]
# Daten formatieren und KMeans
n_cluster = 3
x = np.array([alpha,beta,gamma,delta,epsilon, iota, theta]).T
contact_points = x[force_indexes]
kmeans = skc.KMeans(n_cluster, max_iter=300, n_init='auto').fit(contact_points)
pred = kmeans.predict(contact_points)
kmeans_center = kmeans.cluster_centers_

timestamp_vis = 0
cluster = []

for i in range(n_cluster):

    cluster = np.where(pred == i)

    # Find indexes (timestamp) of datapoints closest to cluster centers
    timestamp_total, closest_point = find_timestamp_to_cluster_center(kmeans_center[i], contact_points)
    timestamp_total = timestamp_total[0][0]
    print(timestamp_total)

    # Create graph for each cluster
    
    force_cluster = ft_norm[force_indexes][cluster]
    tendon_force_cluster = sum_tendon_force[force_indexes][cluster]

    # Find point in threshold, clustered data that is closest to the cluster_center
    timestamp_cluster, _ = find_timestamp_to_cluster_center(kmeans_center[i], cluster)
    closest_point_in_ft_force = force_cluster[timestamp_cluster[0][0]] # type: ignore
    index_cluster_center = np.where(force_cluster == closest_point_in_ft_force)  # Extract the first element of the tuple

    center_str = str(timestamp_total)
    tf.plot_grip_force_fit_exp_show_center(tendon_force_cluster,force_cluster, index_cluster_center, center_str)
# %%