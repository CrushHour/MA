import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP")

    y = Tracker_DAU_DIP.track_traj_opt[:,4]
    dy = np.gradient(y)

    show_plots = False

    if show_plots:
        print('mean:',np.mean(dy))
        print('std:',np.std(dy))
        print('max:',np.max(dy))
        print('min:', np.min(dy))
        print('median:', np.median(dy))
        print('---')
        print('mean + 1 std:',np.mean(dy)+np.std(dy))
        print('mean - 1 std:',np.mean(dy)-np.std(dy))
        print('mean + 2 std:',np.mean(dy)+2*np.std(dy))
        print('mean - 2 std:',np.mean(dy)-2*np.std(dy))

        plt.hist(dy, bins=1000)
        plt.show()
        plt.close()

    n_std = 4

    indicies = np.where(dy > np.mean(dy)+n_std*np.std(dy))
    indicies2 = np.where(dy < np.mean(dy)-n_std*np.std(dy))
    print(indicies[0].shape)
    print(indicies2[0].shape)

    # overwrite values where index is in indicies
    x_new = np.copy(Tracker_DAU_DIP.track_traj_opt[:,4])
    for i in indicies:
        x_new[i] = Tracker_DAU_DIP.track_traj_opt[i-1,4]
    for i in indicies2:
        x_new[i] = Tracker_DAU_DIP.track_traj_opt[i-1,4]

    plt.plot(Tracker_DAU_DIP.track_traj_opt[:,4], label='original', alpha=0.5, color='blue')
    plt.plot(x_new, label='corrected', alpha=0.5, color='red')
    plt.legend()
    plt.show()