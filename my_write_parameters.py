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


def build_parameters(data, trackers):
    # root = get_vec(data, 2) / 100.0
    # pip_vec = (get_vec(data, 1) - get_vec(data, 2)) / 100.0
    # dp_vec = (get_vec(data, 0) - get_vec(data, 2)) / 100.0

    parameters = {
        "offset": "0 0 0",
        "mcp": {
            "range": "-2 1",
        },
        "pip": {
            "range": "0 2.5",
        },
        "dp": {
            "range": "0 1",
        },
    }
    for file in data:
        print(data, file)
        json_data = json.load(open(data[file]))
        center = (
            np.array(json_data["markups"][0]["controlPoints"][0]["position"])
            + np.array(json_data["markups"][0]["controlPoints"][1]["position"])
        ) / 2
        parameters[file]["axis"] = "1 0.1 0.1"
        parameters[file]["pos"] = center

    parameters["offset"] = parameters["mcp"]["pos"]
    parameters["pip"]["pos"] = vec_to_string(parameters["pip"]["pos"] - parameters["offset"])
    parameters["dp"]["pos"] = vec_to_string(parameters["dp"]["pos"] - parameters["offset"])
    parameters["offset"] = vec_to_string(parameters["offset"])
    parameters["mcp"]["pos"] = "0 0 0"
    return parameters


files = {
    "dau": {
        "dp": "Data/Slicer3D/Joints/DAU_A_DIP.mrk.json",
        "pip": "Data/Slicer3D/Joints/DAU_A_PIP.mrk.json",
        "mcp": "Data/Slicer3D/Joints/DAU_A_MCP.mrk.json",

    },
    "zf": {
        "dip": "Data/Slicer3D/Joints/ZF_A_DIP.mrk.json",
        "pip": "Data/Slicer3D/Joints/ZF_A_PIP.mrk.json",
        "mcp": "Data/Slicer3D/Joints/ZF_A_MCP.mrk.json",

    },
}

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

parameters = dict()
for name in files:
    trackers = dict()
    # for tracker_name in tracker_files[name]:
    #     tracker_data = load_fcsv(tracker_files[name][tracker_name])
    #     trackers[tracker_name] = [
    #         get_vec(tracker_data, 0) / 100.0,
    #         get_vec(tracker_data, 1) / 100.0,
    #         get_vec(tracker_data, 2) / 100.0,
    #     ]
    # data = load_json(files[name])
    parameters[name] = build_parameters(files[name], trackers)


#with open("generated_parameters.yaml", "w") as outfile:
#    yaml.dump(parameters, outfile)
