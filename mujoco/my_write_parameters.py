import csv
import yaml
import json
import numpy as np
import transformation_functions as tf


def load_fcsv(filename):
    labels = []
    data = dict()
    with open(filename) as csvfile:
        label_reached = False
        while not label_reached:
            line = csvfile.readline()
            label_trigger = "# columns = "
            if line.startswith(label_trigger):
                label_reached = True
                labels = line.replace(label_trigger, "").split(",")
                for label in labels:
                    data[label] = []

        spamreader = csv.reader(csvfile, delimiter=",", quotechar="#")
        for row in spamreader:
            for label, value in zip(labels, row):
                data[label].append(value)
    return data


def get_vec(data, index):
    vec = []
    for label in ["x", "y", "z"]:
        vec.append(float(data[label][index]))
    return np.array(vec)


def vec_to_string(vec):
    vec = map(str, vec)
    return " ".join(vec)


def build_parameters(data = [[1,0,0,0],[0,0,0]]):
    parameters = dict()
    parameters['quat'] = vec_to_string(data[0])
    parameters['pos'] = vec_to_string(data[1])
    return parameters




# tracker_files = {
#     "dau": {
#         "mcp": "landmarks/TRACKER_DAU_MCP.fcsv",
#         "pip": "landmarks/TRACKER_DAU_PIP.fcsv",
#     },
#     "index": {
#         "mcp": "landmarks/TRACKER_ZEI_MCP.fcsv",
#         "pip": "landmarks/TRACKER_DAU_PIP.fcsv",
#     },
# }




