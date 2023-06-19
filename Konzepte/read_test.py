# %%

import numpy as np
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ipywidgets import widgets
import os
from pyquaternion import Quaternion

"""
Strecker 1 zeigerfinger: 5
Strecker 2 zeigerfinger: 4

Beuger 1 zeigerfinger: 1
Beuger 2 zeigerfinger: 3

Strecker 1 Daumen: 6
Strecker 2 Daumen: 7

Daumen Abspreitzer: 2
Daumen Beuger: 0


Id
ZF PP : 1007
ZF DP : 1008

DAUMEN DP : 1009
DAUMEN MC : 1010

empty: 1011
"""


@dataclass
class FingerAssignment:
    zf_strecker_1: int
    zf_strecker_2: int

    zf_beuger_1: int
    zf_beuger_2: int

    daumen_strecker_1: int
    daumen_strecker_2: int

    daumen_spreitzer: int
    daumen_beuger: int


@dataclass
class RigidBodyAssignment:
    zf_pp: int
    zf_dp: int
    daumen_dp: int
    daumen_mc: int
    force_torque: int


class TestEvaluator():

    def __init__(self, finger_a: FingerAssignment, body_a: RigidBodyAssignment, name='pincer_highscore.json', path='./'):

        # read the json
        self.name = f'{path}/{name}'
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        # extract the main vectors
        self.obs = data['observation']
        self.act = data['action']
        self.time = [t / 1000 for t in data['time']]

        self.body_a = body_a
        self.finger_a = finger_a
        self.assign_rigid_bodies()

        self.clear_quaternions()

    def assign_rigid_bodies(self):
        """assign the rigid bodies acco to the names"""
        for attribute in dir(self.body_a):

            if ('_' in attribute[0]) == False:
                print(attribute)
                setattr(self, attribute,
                        self.obs['rigid_bodies'][getattr(self.body_a, attribute)])
                self.inverse_quaternions(attribute)

    def inverse_quaternions(self, call_name):
        """inverse the quaternions"""
        rigid_b = getattr(self, call_name)

        prev = np.array([1, 0, 0, 0])
        # watch signums
        for i, ele in enumerate(zip(rigid_b['qw'], rigid_b['qx'], rigid_b['qy'], rigid_b['qz'])):
            w, x, y, z = ele
            cur = np.array([w, x, y, z])

            if np.linalg.norm(prev-cur) < np.linalg.norm(prev + cur):

                rigid_b['qw'][i] = -w
                rigid_b['qx'][i] = -x
                rigid_b['qy'][i] = -y
                rigid_b['qz'][i] = -z

            prev = np.array([w, x, y, z])

        # catch inverse flips
    def inverse_quat2(self, call_name, eps=0.1):
        """inverse the quaternions and their jumps"""
        rigid_b = getattr(self, call_name)
        qx = []
        qy = []
        qz = []
        qw = []

        # detect inverse jumpy
        for (w, x, y, z) in zip(rigid_b['qw'], rigid_b['qx'], rigid_b['qy'], rigid_b['qz']):
            q = Quaternion(w, x, y, z)


            if q.w < 0:
                q = -q


            qx.append(q.x)
            qy.append(q.y)
            qz.append(q.z)
            qw.append(q.w)

        qx_new = []
        qy_new = []
        qz_new = []
        qw_new = []

        qx_prev = qx[0]
        qy_prev = qy[0]
        qz_prev = qz[0]
        qw_prev = qw[0]

        # detect remaining jumps
        for (x, y, z, w) in zip(qx, qy, qz, qw):
            if abs(abs(qw_prev / w) - 1) > eps:
                qx_new.append(qx_prev)
                qy_new.append(qy_prev)
                qz_new.append(qz_prev)
                qw_new.append(qw_prev)
            else:
                qx_new.append(x)
                qy_new.append(y)
                qz_new.append(z)
                qw_new.append(w)

                qx_prev = x
                qy_prev = y
                qz_prev = z
                qw_prev = w

        # finally reassign the quaternions
        setattr(self, call_name, {'qw': qw_new, 'qx': qx_new, 'qy': qy_new,
                                  'qz': qz_new, 'x': rigid_b['x'], 'y': rigid_b['y'], 'z': rigid_b['z']})

    def clear_quaternions(self):
        """clear the quaternions and their jumps"""
        for attribute in dir(self.body_a):
            if ('_' in attribute[0]) == False:
                self.inverse_quat2(attribute)

    def plot_rigid_bodies(self, call_name, pl_pos=False, lim=0):
        """make a simple plot, containing the general quaternion and position data"""
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1) if pl_pos else None
        plt.title(call_name)
        rigid_b = getattr(self, call_name)

        if lim == 0:
            lim = len(self.time)

        plt.plot(self.time[:lim], rigid_b['qx'][:lim], label='qx')
        plt.plot(self.time[:lim], rigid_b['qy'][:lim], label='qy')
        plt.plot(self.time[:lim], rigid_b['qz'][:lim], label='qz')
        plt.plot(self.time[:lim], rigid_b['qw'][:lim], label='qw')
        plt.grid()
        plt.ylabel('quaternion')
        plt.legend()

        if pl_pos:
            plt.subplot(2, 1, 2)
            plt.plot(self.time[:lim], rigid_b['x'][:lim], label='x')
            plt.plot(self.time[:lim], rigid_b['y'][:lim], label='y')
            plt.plot(self.time[:lim], rigid_b['z'][:lim], label='z')
            plt.grid()
            plt.legend()
            plt.ylabel('position')

        plt.xlabel('time [s]')



#data = TestEvaluator(finger_a, body_a)


def read_all_files(idx, lim=5000, path='./'):
    filename = testfiles[idx]
    print(filename)

    finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
    body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
    data = TestEvaluator(finger_a, body_a, name=filename, path=path)

    fx = [data.obs['force_torques'][i]['fx']
          for i in range(len(data.obs['force_torques']))][0]
    fy = [data.obs['force_torques'][i]['fy']
          for i in range(len(data.obs['force_torques']))][0]
    fz = [data.obs['force_torques'][i]['fz']
          for i in range(len(data.obs['force_torques']))][0]

    f_all = [np.sqrt(fxi**2 + fyi**2 + fzi**2)
             for fxi, fyi, fzi in zip(fx, fy, fz)]

    tlim = 4400
    fs = 12

    name_list = [
        'Extensor pollicis brevis',
        'Extensor pollicis longus',
        'Abductor pollicis longus',
        'Flexor pollicis longus',
        'Extensor digitorum',
        'Extensor indicis',
        'Flexor digitorum superficialis',
        'Flexor digitorum profundus',
    ]
    id_list = [7, 5, 0, 1, 4, 6, 2, 3]

    sign_list = [
        '-',
        '-.',
        '--',
        ':',
    ]

    motor_forces = {}

    for loc_name, idx in zip(name_list, id_list):
        motor_forces[loc_name] = []
        for i in range(len(data.act)):
            motor_forces[loc_name].append(data.act[i][0][idx])

    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)

    for force, sty in zip(name_list[:4], sign_list):
        plt.plot(data.time[tmin:tlim], motor_forces[force]
                 [tmin:tlim], sty, label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces thumb [N]', fontsize=fs)
    plt.ylim([-5, 40])

    plt.subplot(3, 1, 2)
    for force, sty in zip(name_list[4:8], sign_list):
        plt.plot(data.time[tmin:tlim], motor_forces[force]
                 [tmin:tlim], sty,  label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces index [N]', fontsize=fs)
    plt.ylim([-5, 40])

    plt.subplot(3, 1, 3)
    plt.plot(data.time[tmin:tlim], f_all[tmin:tlim])
    plt.grid()
    plt.xlabel('time [s]', fontsize=fs)
    plt.ylabel('pincer force [N]', fontsize=fs)
    plt.ylim([-5, 40])


    #data.plot_rigid_bodies('force_torque')
    #data.plot_rigid_bodies('zf_pp', lim=lim)
    #data.plot_rigid_bodies('zf_dp', lim=lim)
    #data.plot_rigid_bodies('daumen_dp', lim=lim)
    #data.plot_rigid_bodies('daumen_mc', lim=lim)

    



def t_filt(arr, t=0.9):
    new_arr = arr.copy()
    for i in range(1, len(arr)):
        new_arr[i] = (1 - t) * arr[i] + t * new_arr[i - 1]
    return new_arr


def plot_forces_highscore(tmin, tlim, fs=12):
    """plot the force data of the highscore"""
    data = TestEvaluator(finger_a, body_a, name='pincer_highscore.json')

    fx = [data.obs['force_torques'][i]['fx']
          for i in range(len(data.obs['force_torques']))][0]
    fy = [data.obs['force_torques'][i]['fy']
          for i in range(len(data.obs['force_torques']))][0]
    fz = [data.obs['force_torques'][i]['fz']
          for i in range(len(data.obs['force_torques']))][0]

    f_all = [np.sqrt(fxi**2 + fyi**2 + fzi**2)
             for fxi, fyi, fzi in zip(fx, fy, fz)]

    tlim = 4400
    fs = 12

    name_list = [
        'Extensor pollicis brevis',
        'Extensor pollicis longus',
        'Abductor pollicis longus',
        'Flexor pollicis longus',
        'Extensor digitorum',
        'Extensor indicis',
        'Flexor digitorum superficialis',
        'Flexor digitorum profundus',
    ]
    id_list = [7, 5, 0, 1, 4, 6, 2, 3]

    sign_list = [
        '-',
        '-.',
        '--',
        ':',
    ]

    motor_forces = {}

    for loc_name, idx in zip(name_list, id_list):
        motor_forces[loc_name] = []
        for i in range(len(data.act)):
            motor_forces[loc_name].append(data.act[i][0][idx])

    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)
    for force, sty in zip(name_list[:4], sign_list):
        plt.plot(data.time[tmin:tlim], motor_forces[force]
                 [tmin:tlim], sty, label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces thumb [N]', fontsize=fs)
    plt.ylim([-5, 40])

    plt.subplot(3, 1, 2)
    for force, sty in zip(name_list[4:8], sign_list):
        plt.plot(data.time[tmin:tlim], motor_forces[force]
                 [tmin:tlim], sty,  label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces index [N]', fontsize=fs)
    plt.ylim([-5, 40])

    plt.subplot(3, 1, 3)
    plt.plot(data.time[tmin:tlim], f_all[tmin:tlim])
    plt.grid()
    plt.xlabel('time [s]', fontsize=fs)
    plt.ylabel('pincer force [N]', fontsize=fs)
    plt.ylim([-5, 40])

    return f_all


def plot_positions(tmin, tlim, fs=12, alp_grid=0.25):
    """plot the position data of the highscore"""
    data = TestEvaluator(finger_a, body_a, name='pincer_highscore.json')

    v_values = [1.7, 3.1, 4.35, 5.8, 7.5, 9.1]


    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.subplot(2, 2, 1)
    plt.title('thumb', fontsize=fs)
    plt.plot(data.time[tmin:tlim], [x - data.daumen_dp['x'][tmin]
                                    for x in data.daumen_dp['x'][tmin:tlim]], label='x')
    plt.plot(data.time[tmin:tlim], [y - data.daumen_dp['y'][tmin]
                                    for y in data.daumen_dp['y'][tmin:tlim]], label='y')
    plt.plot(data.time[tmin:tlim], [z - data.daumen_dp['z'][tmin]
                                    for z in data.daumen_dp['z'][tmin:tlim]], label='z')
    
    for x_val in v_values:
        plt.vlines(x_val, -1, 1, linestyles='dashed', colors='k', linewidth=1, alpha=0.5)

    plt.grid(alpha=alp_grid)
    plt.ylim([-0.1, 0.15])
    plt.ylabel('position [m]', fontsize=fs)

    plt.subplot(2, 2, 2)
    plt.title('index', fontsize=fs)
    plt.plot(data.time[tmin:tlim], [x - data.zf_dp['x'][tmin]
                                    for x in data.zf_dp['x'][tmin:tlim]], label='x')
    plt.plot(data.time[tmin:tlim], [y - data.zf_dp['y'][tmin]
                                    for y in data.zf_dp['y'][tmin:tlim]], label='y')
    plt.plot(data.time[tmin:tlim], [z - data.zf_dp['z'][tmin]
                                    for z in data.zf_dp['z'][tmin:tlim]], label='z')
    plt.grid(alpha=alp_grid)
    plt.ylim([-0.1, 0.15])
    

    plt.subplot(2, 2, 3)
    plt.plot(data.time[tmin:tlim], data.daumen_dp['qx'][tmin:tlim], label='qx')
    plt.plot(data.time[tmin:tlim], data.daumen_dp['qy'][tmin:tlim], label='qy')
    plt.plot(data.time[tmin:tlim], data.daumen_dp['qz'][tmin:tlim], label='qz')
    plt.plot(data.time[tmin:tlim], data.daumen_dp['qw'][tmin:tlim], label='qw')

    for x_val in v_values:
        plt.vlines(x_val, -1, 1, linestyles='dashed', colors='k', linewidth=1, alpha=0.5)

    plt.grid(alpha=alp_grid)
    plt.ylim([-1, 1])
    plt.ylabel('quaternion', fontsize=fs)
    plt.xlabel('time [s]', fontsize=fs)

    plt.subplot(2, 2, 4)
    plt.plot(data.time[tmin:tlim], data.zf_dp['qx'][tmin:tlim], label='qx')
    plt.plot(data.time[tmin:tlim], data.zf_dp['qy'][tmin:tlim], label='qy')
    plt.plot(data.time[tmin:tlim], data.zf_dp['qz'][tmin:tlim], label='qz')
    plt.plot(data.time[tmin:tlim], data.zf_dp['qw'][tmin:tlim], label='qw')
    plt.grid(alpha=alp_grid)
    plt.ylim([-1, 1])
    plt.xlabel('time [s]', fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs)


# %%



if __name__ == '__main__':
    finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
    body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
    tmin = 500
    tlim = 4400
    fs = 12
    
    #f_all = plot_forces_highscore(tmin, tlim, fs)
    #plot_positions(tmin, tlim, fs)
    data = TestEvaluator(finger_a, body_a, name='2023_01_30_18_46_11.json', path='./Data/test_01_30')
    testfiles = os.listdir('./Data/test_01_30')
    testfiles.sort()
    testfiles = testfiles[:-1]
    
    idx_test = widgets.IntSlider(min=0, max=len(testfiles), value=0)
    widgets.interact(read_all_files, idx=idx_test, path='./Data/test_01_30')


# %%

    data = TestEvaluator(finger_a, body_a, name='2023_01_30_21_50_38.json', path='./Data/test_01_30')
    # %%
    data.act[0][0]
    # %%
    len(data.act)
    # %%
    fx = [data.obs['force_torques'][i]['fx']
            for i in range(len(data.obs['force_torques']))][0]
    fy = [data.obs['force_torques'][i]['fy']
            for i in range(len(data.obs['force_torques']))][0]
    fz = [data.obs['force_torques'][i]['fz']
            for i in range(len(data.obs['force_torques']))][0]

    f_all = [np.sqrt(fxi**2 + fyi**2 + fzi**2)
                for fxi, fyi, fzi in zip(fx, fy, fz)]

    fs = 12

    name_list = [
        'Extensor pollicis brevis',
        'Extensor pollicis longus',
        'Abductor pollicis longus',
        'Flexor pollicis longus',
        'Extensor digitorum',
        'Extensor indicis',
        'Flexor digitorum superficialis',
        'Flexor digitorum profundus',
    ]
    id_list = [7, 5, 0, 1, 4, 6, 2, 3]

    sign_list = [
        '-',
        '-.',
        '--',
        ':',
    ]

    motor_forces = {}

    for loc_name, idx in zip(name_list, id_list):
        motor_forces[loc_name] = []
        for i in range(len(data.act)):
            motor_forces[loc_name].append(data.act[i][0][idx])

    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)

    for force, sty in zip(name_list[:4], sign_list):
        plt.plot(motor_forces[force], sty, label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces thumb [N]', fontsize=fs)
    #plt.ylim([-5, 40])

    plt.subplot(3, 1, 2)
    for force, sty in zip(name_list[4:8], sign_list):
        plt.plot(motor_forces[force]
                    , sty,  label=force)
    plt.grid()
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('forces index [N]', fontsize=fs)
    #plt.ylim([-5, 40])

    plt.subplot(3, 1, 3)
    plt.plot(f_all)
    plt.grid()
    plt.xlabel('time [s]', fontsize=fs)
    plt.ylabel('pincer force [N]', fontsize=fs)
    #plt.ylim([-5, 40])
    # %%
    motor_forces
    # %%
