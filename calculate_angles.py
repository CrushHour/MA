# %% Import
import sys
import os
import time
sys.path.append('./mujoco')
import transformation_functions as tf
import calibrate_sensor as cs
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import my_write_parameters as mwp
import my_model as mwj
import yaml
from pyquaternion import Quaternion
import importlib
from scipy.optimize import curve_fit


importlib.reload(tf)

# from scipy.spatial.transform import Rotation as Rot

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def create_test_metadata_from_path(path):
    test_metadata = dict()
    test_metadata['name'] = path.split('/')[-1][:-5]
    test_metadata['path'] = path
    test_metadata['marker_IDs'] = ["",""]
    test_metadata['type'] = path[-4:]
    test_metadata['length'] = len(tf.get_json(path)['time'])
    return test_metadata

# %%Definition der Pfade
hand_metadata = tf.get_json("hand_metadata.json")

test_number = 2

xtick_range = 0
start = 0
end = 0

if test_number == 0:
    test_metadata = tf.get_json("test_metadata.json")["Take 2023-01-31 06.11.42 PM.csv"]
    xtick_range = 5
elif test_number == 1:
    test_metadata = tf.get_json("test_metadata.json")["optitrack-20230130-234800.json"]
elif test_number == 2:
    test_metadata = tf.get_json("test_metadata.json")["2023_01_31_18_12_48.json"]
    xtick_range = 1000
    start = 1400
    end = -1
else:
    # test_metadata = tf.get_json("test_metadata.json")["2023_01_31_18_12_48.json"]
    #   test_metadata = tf.get_json('test_metadata.json')['2023_01_31_18_10_36.json']
    #   test_metadata = tf.get_json('test_metadata.json')['2023_01_31_18_08_11.json']
    test_metadata = tf.get_json("test_metadata.json")["2023_01_31_00_47_54.json"]
    #   test_metadata = tf.get_json('test_metadata.json')['2023_01_31_00_42_47.json']
    #   test_metadata = tf.get_json('test_metadata.json')['2023_01_31_00_30_44.json']
    #   test_metadata = tf.get_json('test_metadata.json')['test']
    xtick_range = 1000
    start = 1100
    # start = 900
    end = start + 370
    # end = 1550
# %%Tracker
# Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error
# [X	        Y	        Z	        W]	        [X	        Y	        Z]

# Definieren der Tracker und Marker als jeweils eine Tracker Klasse
Tracker_ZF_DIP = tf.tracker_bone("ZF_DIP", test_path=test_metadata["path"])
Tracker_ZF_midhand = tf.tracker_bone("ZF_midhand", test_path=test_metadata["path"])
Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP", test_path=test_metadata["path"])
Tracker_DAU_MCP = tf.tracker_bone("DAU_MCP", test_path=test_metadata["path"])
# Tracker_FT = tf.tracker_bone('FT',test_path=test_metadata['path'])

Marker_DAU = tf.marker_bone(
    finger_name="DAU_PIP",
    test_path=test_metadata["path"],
    init_marker_ID=test_metadata["marker_IDs"][1],
)
Marker_ZF_intermedial = tf.marker_bone(
    finger_name="ZF_PIP",
    test_path=test_metadata["path"],
    init_marker_ID=test_metadata["marker_IDs"][0],
)
ZF_MCP = tf.marker_bone(
    finger_name="ZF_MCP", test_path=test_metadata["path"], init_marker_ID=""
)

# %% Berechnung der Markerpositionen im CT


def construct_marker_rot(opt_info, ct_info):
    """Do kabsch with points from different sources. Points must be in the same order."""
    T = Tracker_DAU_DIP.kabsch(ct_info, opt_info)
    return T


for t in tqdm(range(len(Marker_DAU.opt_marker_trace))):
    Marker_DAU.T_opt_ct[t] = construct_marker_rot(
        [
            Tracker_DAU_DIP.T_proxi_innen_opt[t, :3, 3],
            Tracker_DAU_DIP.T_proxi_aussen_opt[t, :3, 3],
            Tracker_DAU_MCP.T_dist_innen_opt[t, :3, 3],
            Tracker_DAU_MCP.T_dist_aussen_opt[t, :3, 3],
        ],
        [
            Tracker_DAU_DIP.T_proxi_innen_CT[:3, 3],
            Tracker_DAU_DIP.T_proxi_aussen_CT[:3, 3],
            Tracker_DAU_MCP.T_dist_innen_CT[:3, 3],
            Tracker_DAU_MCP.T_dist_aussen_CT[:3, 3],
        ],
    )
    Marker_DAU.update_joints(t)

    # Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot([Marker_ZF_intermedial.opt_marker_trace[t],Tracker_ZF_DIP.T_proxi_innen_opt[t,:3,3],Tracker_ZF_DIP.T_proxi_aussen_opt[t,:3,3]], \
    #                                                      [np.array(Marker_ZF_intermedial.marker_pos_ct[0]), Tracker_ZF_DIP.T_proxi_innen_CT[:3,3], Tracker_ZF_DIP.T_proxi_aussen_CT[:3,3]])
    
    
    
    Marker_ZF_intermedial.T_opt_ct[t] = Tracker_ZF_DIP.T_opt_ct[t]

    # update loop auf basis aller bekannten Punkte
    for j in range(3):
    
        Marker_ZF_intermedial.update_joints(t)

        ZF_MCP.T_opt_ct[t] = construct_marker_rot(
            [
                Tracker_ZF_midhand.T_dist_innen_opt[t, :3, 3],
                Tracker_ZF_midhand.T_dist_aussen_opt[t, :3, 3],
                Marker_ZF_intermedial.T_proxi_opt[t, 0, :3, 3],
                Marker_ZF_intermedial.T_proxi_opt[t, 1, :3, 3],
            ],
            [
                Tracker_ZF_midhand.T_dist_innen_CT[:3, 3],
                Tracker_ZF_midhand.T_dist_aussen_CT[:3, 3],
                Marker_ZF_intermedial.T_proxi_CT[0, :3, 3],
                Marker_ZF_intermedial.T_proxi_CT[1, :3, 3],
            ],
        )
        ZF_MCP.update_joints(t)

        Marker_ZF_intermedial.T_opt_ct[t] = construct_marker_rot(
            [
                Tracker_ZF_DIP.T_proxi_innen_opt[t, :3, 3],
                Tracker_ZF_DIP.T_proxi_aussen_opt[t, :3, 3],
                ZF_MCP.T_dist_opt[t, 0, :3, 3],
                ZF_MCP.T_dist_opt[t, 1, :3, 3],
            ],
            [
                Tracker_ZF_DIP.T_proxi_innen_CT[:3, 3],
                Tracker_ZF_DIP.T_proxi_aussen_CT[:3, 3],
                ZF_MCP.T_dist_CT[0, :3, 3],
                ZF_MCP.T_dist_CT[1, :3, 3],
            ],
        )


# %% Build mujoco parameters
# i = 4553
# i = 5831
i = 0

parameters = {"zf": dict(), "dau": dict()}

# STL
parameters["zf"]["dip"] = mwp.build_parameters(
    [
        Quaternion(matrix=Tracker_ZF_DIP.T_opt_ct[i, :3, :3]),
        Tracker_ZF_DIP.T_opt_ct[i, :3, 3],
    ]
)
parameters["zf"]["pip"] = mwp.build_parameters(
    [
        Quaternion(matrix=Marker_ZF_intermedial.T_opt_ct[i, :3, :3]),
        Marker_ZF_intermedial.T_opt_ct[i, :3, 3],
    ]
)
parameters["zf"]["mcp"] = mwp.build_parameters(
    [Quaternion(matrix=ZF_MCP.T_opt_ct[i, :3, :3]), ZF_MCP.T_opt_ct[i, :3, 3]]
)
parameters["zf"]["midhand"] = mwp.build_parameters(
    [
        Quaternion(matrix=Tracker_ZF_midhand.T_opt_ct[i, :3, :3]),
        Tracker_ZF_midhand.T_opt_ct[i, :3, 3],
    ]
)

# green, red, yellow, white, blue balls
parameters["zf"]["dip_joint_aussen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_ZF_DIP.T_proxi_aussen_opt[i, :3, 3]]
)
parameters["zf"]["dip_joint_innen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_ZF_DIP.T_proxi_innen_opt[i, :3, 3]]
)
parameters["zf"]["pip_joint_aussen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Marker_ZF_intermedial.T_proxi_opt[i, 1, :3, 3]]
)
parameters["zf"]["pip_joint_innen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Marker_ZF_intermedial.T_proxi_opt[i, 0, :3, 3]]
)
parameters["zf"]["mcp_joint_aussen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_ZF_midhand.T_dist_aussen_opt[i, :3, 3]]
)
parameters["zf"]["mcp_joint_innen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_ZF_midhand.T_dist_innen_opt[i, :3, 3]]
)
parameters["zf"]["pip_marker"] = mwp.build_parameters(
    [[1, 0, 0, 0], Marker_ZF_intermedial.opt_marker_trace[i]]
)

# STL
parameters["dau"]["dip"] = mwp.build_parameters(
    [
        Quaternion(matrix=Tracker_DAU_DIP.T_opt_ct[i, :3, :3]),
        Tracker_DAU_DIP.T_opt_ct[i, :3, 3],
    ]
)
parameters["dau"]["pip"] = mwp.build_parameters(
    [Quaternion(matrix=Marker_DAU.T_opt_ct[i, :3, :3]), Marker_DAU.T_opt_ct[i, :3, 3]]
)
parameters["dau"]["mcp"] = mwp.build_parameters(
    [
        Quaternion(matrix=Tracker_DAU_MCP.T_opt_ct[i, :3, :3]),
        Tracker_DAU_MCP.T_opt_ct[i, :3, 3],
    ]
)

# green, yellow balls
parameters["dau"]["pip_joint"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_DAU_DIP.T_proxi_opt[i, :3, 3]]
)
parameters["dau"]["mcp_joint"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_DAU_MCP.T_dist_opt[i, :3, 3]]
)
parameters["dau"]["mcp_joint2"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_DAU_MCP.T_proxi_opt[i, :3, 3]]
)
parameters["dau"]["pip_marker"] = mwp.build_parameters(
    [[1, 0, 0, 0], Marker_DAU.opt_marker_trace[i]]
)
parameters["dau"]["mcp_joint_aussen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_DAU_MCP.T_proxi_aussen_opt[i, :3, 3]]
)
parameters["dau"]["mcp_joint_innen"] = mwp.build_parameters(
    [[1, 0, 0, 0], Tracker_DAU_MCP.T_proxi_innen_opt[i, :3, 3]]
)


with open("./mujoco/generated_parameters.yaml", "w") as outfile:
    yaml.dump(parameters, outfile)

#model = mwj.MujocoFingerModel(
#    "./mujoco/my_tendom_finger_template.xml", "./mujoco/generated_parameters.yaml"
#)
print("Model updated!")

# %% plotten von delta und epsilon


if __name__ == "__main__":
    save_plots = False

    alpha = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    beta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    gamma = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    delta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    epsilon = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    zeta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    eta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    theta = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    iota = np.zeros(len(Marker_ZF_intermedial.opt_marker_trace))
    axis = np.zeros((len(Marker_ZF_intermedial.opt_marker_trace), 3))
    q = []

    # vZF_PIP = np.subtract(np.mean(Marker_ZF_intermedial.T_dist_opt[i,:,:3,3],axis=0),np.mean(Marker_ZF_intermedial.T_proxi_opt[i,:,:3,3],axis=0))
    # vZF_MCP = np.subtract(np.mean(ZF_MCP.T_dist_opt[i,:,:3,3],axis=0),np.mean(ZF_MCP.T_proxi_opt[i,:,:3,3],axis=0))

    degree = True

    for i in range(len(Marker_ZF_intermedial.opt_marker_trace)):
        if degree == True:
            trans = 180 / np.pi
        else:
            trans = 1
        vZF_DIP = np.subtract(
            Tracker_ZF_DIP.T_dist_opt[i, :3, 3], Tracker_ZF_DIP.T_proxi_opt[i, :3, 3]
        )
        vZF_PIP = np.subtract(
            np.mean(Marker_ZF_intermedial.T_dist_opt[i, :, :3, 3], axis=0),
            np.mean(Marker_ZF_intermedial.T_proxi_opt[i, :, :3, 3], axis=0),
        )
        vZF_MCP = np.subtract(
            np.mean(ZF_MCP.T_dist_opt[i, :, :3, 3], axis=0),
            np.mean(ZF_MCP.T_proxi_opt[i, :, :3, 3], axis=0),
        )

        vDAU_PIP = np.subtract(
            np.mean(Marker_DAU.T_dist_opt[i, :, :3, 3], axis=0),
            np.mean(Marker_DAU.T_proxi_opt[i, :, :3, 3], axis=0),
        )

        # calculate angles in ZF joints
        alpha[i] = tf.angle_between(Tracker_ZF_DIP.v_opt[i], vZF_PIP) * trans
        beta[i] = -tf.angle_between(vZF_PIP, vZF_MCP) * trans
        gamma[i] = tf.angle_between(vZF_MCP, Tracker_ZF_midhand.v_opt[i]) * trans

        # calculate angeles in DAU joints
        delta[i] = tf.angle_between(Tracker_DAU_DIP.v_opt[i], vDAU_PIP) * trans
        epsilon[i] = tf.angle_between(vDAU_PIP, Tracker_DAU_MCP.v_opt[i]) * trans

        # axis is defined so that the positive (counterclockwise) angle direction is to the inside of the hand
        axis[i] = np.subtract(
            Tracker_DAU_MCP.T_proxi_aussen_opt[i, :3, 3],
            Tracker_DAU_MCP.T_proxi_innen_opt[i, :3, 3],
        )
        axisp = np.cross(axis[i], Tracker_DAU_MCP.v_opt[i])
        iota[i] = (
            tf.angle_projectet(
                v1=Tracker_DAU_MCP.v_opt[0], v2=Tracker_DAU_MCP.v_opt[i], normal=axis[i]
            )
            * trans
        )
        theta[i] = (
            tf.angle_projectet(
                v1=Tracker_DAU_MCP.v_opt[0], v2=Tracker_DAU_MCP.v_opt[i], normal=axisp
            )
            * trans
        )
    
    

    alpha = tf.interpolate_1d(alpha)
    alpha = Tracker_DAU_DIP.delete_outliers(alpha)
    alpha = tf.interpolate_1d(alpha)

    # catch jumping alphas
    alpha = np.abs(alpha)

    beta = tf.interpolate_1d(beta)
    beta = Tracker_DAU_DIP.delete_outliers(beta, 1.5)
    beta = tf.interpolate_1d(beta)
    gamma = tf.interpolate_1d(gamma)
    gamma = Tracker_DAU_DIP.delete_outliers(gamma)
    gamma = tf.interpolate_1d(gamma)
    delta = tf.interpolate_1d(delta)
    delta = Tracker_DAU_DIP.delete_outliers_local(
        delta, 2, 300
    )  # tune for csv: 2.5, 1500
    delta = tf.interpolate_1d(delta)
    epsilon = tf.interpolate_1d(epsilon)
    epsilon = Tracker_DAU_DIP.delete_outliers_local(
        epsilon, 1.5, 300
    )  # tune for csv: 1.5, 1000
    epsilon = tf.interpolate_1d(epsilon)
    tf.plot_angles(
        [delta[start:end], epsilon[start:end]],
        Tracker_DAU_DIP.time[start:end],
        xtick_range,
        ["delta (DIP)", "epsilon (PIP)"],
        "Angles in Thumb joints",
        save_plots=False,
    )
    tf.plot_angles(
        [alpha[start:end], beta[start:end], gamma[start:end]],
        Tracker_DAU_DIP.time[start:end],
        xtick_range,
        ["alpha (DIP)", "beta (PIP)", "gamma (MCP)"],
        "Angles in Index finger joints",
        save_plots=False,
    )
    # tf.plot_angles([iota[start:end], iota2[start:end]], Tracker_DAU_DIP.time[start:end], xtick_range, ['between vectors', 'around specified axis'], 'Angles in Thumb MCP joint to midhand', save_plots=False)
    tf.plot_angles(
        [theta[start:end]],
        Tracker_DAU_DIP.time[start:end],
        xtick_range,
        ["perpendicular around specified axis"],
        "Angles in Thumb MCP joint to midhand",
        save_plots=False,
    )

    if test_number > 1:
        data = tf.get_json(test_metadata["path"])
        sensor_data = cs.arrange_sensor_data(data)
        # calibrate sensor data
        for i in range(len(sensor_data)):
            sensor_data[i] = cs.apply_calibration(
                sensor_data[i], i, calibration_file="calibration_parameters_long.json"
            )
        thumb_flexor = [sensor_data[i] for i in [3]]  # Beuger
        thumb_extensor = [sensor_data[i] for i in [0, 5, 7]]  # Strecker
        index_flexor = [sensor_data[i] for i in [1, 2]]
        index_extensor = [
            sensor_data[i] for i in [4]
        ]  # 6 hat sich das Offset verändert
        tf.plot_analogs(test_metadata["path"])
        tf.plot_analogs_angles(
            angles=[alpha, beta, gamma],
            flexor=index_flexor,
            extensor=index_extensor,
            time=Tracker_DAU_DIP.time,
            step_size=xtick_range,
            start=start,
            end=end,
            legend1=["DIP index", "PIP index", "MCP index"],
            legend2=["Flexor super", "Flexor profundus"],
            legend3=["Extensor digitorum"],
            title="Angles in index finger joints",
            save_plots=False,
        )
        tf.plot_analogs_angles(
            angles=[delta, epsilon, iota, theta],
            flexor=thumb_flexor,
            extensor=thumb_extensor,
            time=Tracker_DAU_DIP.time,
            step_size=xtick_range,
            start=start,
            end=end,
            legend1=["IP thumb", "MCP thumb", "CMC flexion", "CMC abduction"],
            legend2=["Flexor pollicis longus"],
            legend3=["Abductor", "Extensor longus", "Extensor brevis"],
            title="Angles in thumb joints",
            save_plots=False,
        )

        # plot force torque sensor data
        ft_norm = np.linalg.norm(
            [
                data["observation"]["force_torques"][0]["fz"],
                data["observation"]["force_torques"][0]["fx"],
                data["observation"]["force_torques"][0]["fy"],
            ],
            axis=0,
        )
        # apply facotr from Sensor data to N
        # 1.5587983484301424 nicht benötigt, da Sensor horizontal ausgerichtet ist.
        ft_norm = ft_norm - 1.5587983484301424
        # minimale Gripforce auf 0 setzen
        ft_norm = ft_norm - np.min(ft_norm)
        ft_norm = ft_norm / 8.998278583527435

        tf.plot_ft_norm(
            ft_norm,
            data["time"],
            xtick_range,
            start=start,
            end=end,
            title="Grip force",
            save_plots=False,
        )

        ft_forces = [
            data["observation"]["force_torques"][0]["fx"],
            data["observation"]["force_torques"][0]["fy"],
            data["observation"]["force_torques"][0]["fz"],
        ]
        ft_torques = [
            data["observation"]["force_torques"][0]["mx"],
            data["observation"]["force_torques"][0]["my"],
            data["observation"]["force_torques"][0]["mz"],
        ]
        tf.plot_ft_splitted(
            ft_forces,
            ft_torques,
            data["time"],
            xtick_range,
            start=start,
            end=end,
            legend1=["fx", "fy", "fz"],
            legend2=["mx", "my", "mz"],
            title="FT-Sensor data",
            save_plots=False,
        )

        # calculate tendon forces by summing up the forces of the flexors and extensors
        # - extensor forces are negative
        # - flexor forces are positive
        sum_tendon_force = (
            thumb_flexor[0]
            - thumb_extensor[0]
            - thumb_extensor[1]
            - thumb_extensor[2]
            + index_flexor[0]
            + index_flexor[1]
            - index_extensor[0]
        )
        # sum_tendon_force -= np.mean(sum_tendon_force)
        print("mean tendon force: ", np.mean(sum_tendon_force))
        # plot tendon forces over ft sensor data
        poly_order = 5
        model = np.poly1d(np.polyfit(sum_tendon_force, ft_norm, poly_order))
        polyline = np.linspace(0, max(sum_tendon_force), 100)
        plt.scatter(sum_tendon_force, ft_norm, s=1, color="#0065bd", label="data")
        plt.plot(
            polyline,
            model(polyline),
            color="#e37222",
            label="model " + str(poly_order) + "-th order",
        )
        plt.ylabel("grip force [N]")
        plt.xlabel("Sum of tendon forces [N]")
        plt.title("Tendon forces vs. Grip Force")
        plt.legend()
        if save_plots:
            plt.savefig("plots/tendon_forces_vs_ft_sensor.png", dpi=1200)
        else:
            plt.show()
        plt.close()

        threshold = 0.3
        poly_order = 5
        mask = ft_norm > threshold
        filtered_ft_norm = np.array(ft_norm[mask])
        filtered_sum_tendon_force = np.array(sum_tendon_force[mask])

        # plt.figure(figsize=(12,12))
        model = np.poly1d(
            np.polyfit(filtered_sum_tendon_force, filtered_ft_norm, poly_order)
        )
        polyline = np.linspace(
            0, max(filtered_sum_tendon_force), 100
        )
        plt.scatter(
            filtered_sum_tendon_force,
            filtered_ft_norm,
            s=1,
            color="#0065bd",
            label="data",
        )
        params, covariance = curve_fit(exp_func, filtered_sum_tendon_force, filtered_ft_norm)
        plt.plot(polyline, exp_func(polyline, *params), 'r--', label='Fitted function')
        plt.grid(alpha=0.25)
        plt.ylabel("grip force [N]")
        plt.xlabel("sum of tendon forces [N]")
        plt.title("Correlation of tendon forces with grip strength")
        plt.legend()
        print(f"Fitted parameters: a = {params[0]:.2f}, b = {params[1]:.2f}, c = {params[2]:.2f}")


# %%
