import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf
import matplotlib.pyplot as plt

Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP")

y = Tracker_DAU_DIP.track_traj_opt[:,4]
dy = np.gradient(y)
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


plt.hist(dy, bins=500)
plt.show()