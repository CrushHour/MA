import csv
import yaml
import numpy as np


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
    root = get_vec(data, 2) / 100.0
    pip_vec = (get_vec(data, 1) - get_vec(data, 2)) / 100.0
    dp_vec = (get_vec(data, 0) - get_vec(data, 2)) / 100.0

    parameters = {
        "offset": vec_to_string(-root),
        "mcp": {
            "axis": "1 0.1 0.1",
            "pos": "0 0 0",
            "range": "-2 1",
            "markers": {
                "a": vec_to_string(trackers["mcp"][0] - root),
                "b": vec_to_string(trackers["mcp"][1] - root),
                "c": vec_to_string(trackers["mcp"][2] - root),
            },
        },
        "pip": {
            "axis": "-1 1 0",
            "pos": vec_to_string(pip_vec),
            "range": "0 2.5",
            "markers": {
                "a": vec_to_string(trackers["pip"][0] - root),
                "b": vec_to_string(trackers["pip"][1] - root),
                "c": vec_to_string(trackers["pip"][2] - root),
            },
        },
        "dp": {
            "axis": "-1 1.1 0.1",
            "pos": vec_to_string(dp_vec),
            "range": "0 1",
        },
    }
    return parameters


files = {
    "dau": "landmarks/DAUMEN.fcsv",
    "index": "landmarks/ZEIGEFINGER.fcsv",
}

tracker_files = {
    "dau": {
        "mcp": "landmarks/TRACKER_DAU_MCP.fcsv",
        "pip": "landmarks/TRACKER_DAU_PIP.fcsv",
    },
    "index": {
        "mcp": "landmarks/TRACKER_ZEI_MCP.fcsv",
        "pip": "landmarks/TRACKER_DAU_PIP.fcsv",
    },
}

parameters = dict()
for name in files:
    trackers = dict()
    for tracker_name in tracker_files[name]:
        tracker_data = load_fcsv(tracker_files[name][tracker_name])
        trackers[tracker_name] = [
            get_vec(tracker_data, 0) / 100.0,
            get_vec(tracker_data, 1) / 100.0,
            get_vec(tracker_data, 2) / 100.0,
        ]
    data = load_fcsv(files[name])
    parameters[name] = build_parameters(data, trackers)


with open("generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)
