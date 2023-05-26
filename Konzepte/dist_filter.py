import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP")

pos = Tracker_DAU_DIP.track_traj_opt[:,4:7]

dis = np.array([])
for i in range(len(pos)-1):
    dis = np.append(dis, np.linalg.norm(np.subtract(pos[i],pos[i+1])))
ddis = np.gradient(dis)

n_std = 7

show_plots = True

if show_plots:
    print('mean:',np.mean(dis))
    print('std:',np.std(dis, axis= 0))
    print('max:',np.max(dis))
    print('min:', np.min(dis))
    print('median:', np.median(dis))
    print('---')
    print('mean + 1 std:',np.mean(dis)+np.std(dis))
    print('mean - 1 std:',np.mean(dis)-np.std(dis))
    print('mean + n std:',np.mean(dis)+n_std*np.std(dis))
    print('mean - n std:',np.mean(dis)-n_std*np.std(dis))
    print('n std:',n_std)

    plt.hist(ddis, bins=1000)
    plt.show()
    plt.close()


x = np.copy(Tracker_DAU_DIP.track_traj_opt[:,4:7])

border_val = np.mean(dis)+n_std*np.std(dis, axis= 0)

def replace_outliers(data, threshold):
    clean_data = []
    last_valid = data[0]
    lst_dis = []

    for i in range(len(data)):
        dis = np.linalg.norm(data[i] - last_valid)
        if dis <= threshold:
            clean_data.append(data[i])
            last_valid = data[i]
        else:
            clean_data.append(last_valid)

        lst_dis.append(dis)

    return np.array(clean_data), np.array(lst_dis)

x_new, lst_dis = replace_outliers(x, border_val)

for i in range(3280,3290):
    print(i, lst_dis[i])


plt.plot(Tracker_DAU_DIP.track_traj_opt[:,4:7], label='original', alpha=0.5, color='grey')
plt.plot(x_new[:,0], label='corrected x', alpha=0.5, color='red')
plt.plot(x_new[:,1], label='corrected y', alpha=0.5, color='green')
plt.plot(x_new[:,2], label='corrected z', alpha=0.5, color='blue')

plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
