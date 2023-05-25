import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt

Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP")

pos = Tracker_DAU_DIP.track_traj_opt[:,4:7]

dis = np.array([])
for i in range(len(pos)-1):
    dis = np.append(dis, np.linalg.norm(np.subtract(pos[i],pos[i+1])))
ddis = np.gradient(dis)

show_plots = False

if show_plots:
    print('mean:',np.mean(ddis))
    print('std:',np.std(ddis))
    print('max:',np.max(ddis))
    print('min:', np.min(ddis))
    print('median:', np.median(ddis))
    print('---')
    print('mean + 1 std:',np.mean(ddis)+np.std(ddis))
    print('mean - 1 std:',np.mean(ddis)-np.std(ddis))
    print('mean + 2 std:',np.mean(ddis)+2*np.std(ddis))
    print('mean - 2 std:',np.mean(ddis)-2*np.std(ddis))

    plt.hist(ddis, bins=1000)
    plt.show()
    plt.close()

n_std = 7

x_new = np.copy(Tracker_DAU_DIP.track_traj_opt[:,4:7])
last_valide = x_new[0]
i_save = 0
border_val = np.mean(dis)+n_std*np.std(dis)

print('---Start---')
for i in range(len(ddis)):

    if np.linalg.norm(np.subtract(last_valide,x_new[i])) > border_val:
        print('too big:',ddis[i])
        x_new[i] = last_valide
    else:
        last_valide = x_new[i]
        i_save = i
print('---Stopp---')


plt.plot(Tracker_DAU_DIP.track_traj_opt[:,4:7], label='original', alpha=0.5, color='grey')
plt.plot(x_new[:,0], label='corrected x', alpha=0.5, color='red')
plt.plot(x_new[:,1], label='corrected y', alpha=0.5, color='green')
plt.plot(x_new[:,2], label='corrected z', alpha=0.5, color='blue')

plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
