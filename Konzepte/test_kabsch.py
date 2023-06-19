import numpy as np
import sys
sys.path.append('./')
import transformation_functions as tf

if __name__ == '__main__':

    Tracker_DAU_DIP = tf.tracker_bone("DAU_DIP")

    artificial = False

    P = np.array(Tracker_DAU_DIP.marker_pos_def)
    Q = np.array(Tracker_DAU_DIP.marker_pos_ct)

    if artificial:
        alpha = 0.5
        R = np.array([[np.cos(alpha), -np.sin(alpha), 0, 1],
                        [np.sin(alpha), np.cos(alpha), 0, 2],
                        [0, 0, 1, 3],
                        [0, 0, 0, 1]])
        Q = np.zeros((5,3))
        for i in range(5):
            Q[i,:] = np.matmul(R, np.append(P[i],1))[:3]

    T_ct_def = Tracker_DAU_DIP.kabsch(P,Q) # P nach Q
    T_def_ct = Tracker_DAU_DIP.invert_T(T_ct_def) # Q nach P

    P_recover = np.zeros((5,3))
    for i in range(5):
        P_recover[i,:] = np.matmul(T_def_ct, np.append(Q[i,:],1))[:3]
        print(np.round(P_recover[i,:],4))
        print(np.round(P[i],4))
        print('error:', np.round(np.linalg.norm(P_recover[i,:]-P[i]),4))
        print('---')

    #[10.7669  8.1443 13.4043]
    #[11.5897  8.2104 13.3727]
    #error: 0.826
    #---
    #[ -6.8478 -14.8418  14.7444]
    #[ -6.7487 -14.1018  14.8163]
    #error: 0.7501
    #---
    #[ 19.1674 -14.7319  -2.5076]
    #[ 19.3674 -15.5818  -2.5396]
    #error: 0.8737
    #---
    #[-23.238   11.3677   3.4289]
    #[-24.4736  10.9393   3.1768]
    #error: 1.3318
    #---
    #[  0.1515  10.0617 -29.0699]
    #[  0.2652  10.5339 -28.8261]
    #error: 0.5434
