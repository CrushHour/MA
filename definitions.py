
# %%
import numpy as np
import matplotlib.pyplot as plt


"""
ZEIGEFINGER
Flexor Digitorum -> Motor 1 (I)
Flexor Digitorum -> Motor 2 (II)

Extensor Digitorum -> Motor (4) (I)
Extensor Indicis -> Motor (6) (II)

"""

beuger_offset = 0.0
# STRECKER
# define the motor, amp and offset
ZEIGEFINGER_STRECKER_1_MOTOR = 4
ZEIGEFINGER_STRECKER_1_AMP = 3
ZEIGEFINGER_STRECKER_1_OFFSET = 4
ZEIGEFINGER_STRECKER_1_PHASE = 0

# define the motor, amp and offset
ZEIGEFINGER_STRECKER_2_MOTOR = 6
ZEIGEFINGER_STRECKER_2_AMP = 5
ZEIGEFINGER_STRECKER_2_OFFSET = 3
ZEIGEFINGER_STRECKER_2_PHASE = 0


# BEUGER
# define the motor, amp and offset
ZEIGEFINGER_BEUGER_1_MOTOR = 1
ZEIGEFINGER_BEUGER_1_AMP = -1.5
ZEIGEFINGER_BEUGER_1_OFFSET = 1.2
ZEIGEFINGER_BEUGER_1_PHASE = beuger_offset

# define the motor, amp and offset
ZEIGEFINGER_BEUGER_2_MOTOR = 2
ZEIGEFINGER_BEUGER_2_AMP = -1.0
ZEIGEFINGER_BEUGER_2_OFFSET = 1.7
ZEIGEFINGER_BEUGER_2_PHASE = beuger_offset


"""
DAUMEN

Flexor Pollicis longus -> Motor 3
Abductor policis longus -> Motor 0
Extensor Policis brevis -> Motor 7 (I)
Extensor Policis longus -> Motor 5 (II)

"""
dt_daumen = 0.6 + 0.1
# STRECKER
# define the motor, amp and offset
DAUMEN_STRECKER_1_MOTOR = 7
DAUMEN_STRECKER_1_AMP = 1
DAUMEN_STRECKER_1_OFFSET = 1.2
DAUMEN_STRECKER_1_PHASE = dt_daumen

# define the motor, amp and offset
DAUMEN_STRECKER_2_MOTOR = 5
DAUMEN_STRECKER_2_AMP = 2.0
DAUMEN_STRECKER_2_OFFSET = 1.2
DAUMEN_STRECKER_2_PHASE = dt_daumen

# ABSPREITZER
# define the motor, amp and offset
DAUMEN_ABSPREITZER_1_MOTOR = 0
DAUMEN_ABSPREITZER_1_AMP = 1.0
DAUMEN_ABSPREITZER_1_OFFSET = 1.2
DAUMEN_ABSPREITZER_1_PHASE = dt_daumen

# BEUGER
DAUMEN_BEUGER_2_MOTOR = 3
DAUMEN_BEUGER_2_AMP = -6
DAUMEN_BEUGER_2_OFFSET = 10.0
DAUMEN_BEUGER_2_PHASE = dt_daumen + beuger_offset


"""
FORM THE OUTPUT ARRAYS
"""
action_off = np.array([[
    # 0    1    2    3    4    5    6    7
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]])

action_off[0][ZEIGEFINGER_STRECKER_1_MOTOR] = ZEIGEFINGER_STRECKER_1_OFFSET
action_off[0][ZEIGEFINGER_STRECKER_2_MOTOR] = ZEIGEFINGER_STRECKER_2_OFFSET

action_off[0][ZEIGEFINGER_BEUGER_1_MOTOR] = ZEIGEFINGER_BEUGER_1_OFFSET
action_off[0][ZEIGEFINGER_BEUGER_2_MOTOR] = ZEIGEFINGER_BEUGER_2_OFFSET

action_off[0][DAUMEN_STRECKER_1_MOTOR] = DAUMEN_STRECKER_1_OFFSET
action_off[0][DAUMEN_STRECKER_2_MOTOR] = DAUMEN_STRECKER_2_OFFSET

action_off[0][DAUMEN_BEUGER_2_MOTOR] = DAUMEN_BEUGER_2_OFFSET
action_off[0][DAUMEN_ABSPREITZER_1_MOTOR] = DAUMEN_ABSPREITZER_1_OFFSET


action_amp = np.array([[
    # 0    1    2    3    4    5    6    7
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]])

action_amp[0][ZEIGEFINGER_STRECKER_1_MOTOR] = ZEIGEFINGER_STRECKER_1_AMP
action_amp[0][ZEIGEFINGER_STRECKER_2_MOTOR] = ZEIGEFINGER_STRECKER_2_AMP

action_amp[0][ZEIGEFINGER_BEUGER_1_MOTOR] = ZEIGEFINGER_BEUGER_1_AMP
action_amp[0][ZEIGEFINGER_BEUGER_2_MOTOR] = ZEIGEFINGER_BEUGER_2_AMP

action_amp[0][DAUMEN_STRECKER_1_MOTOR] = DAUMEN_STRECKER_1_AMP
action_amp[0][DAUMEN_STRECKER_2_MOTOR] = DAUMEN_STRECKER_2_AMP

action_amp[0][DAUMEN_BEUGER_2_MOTOR] = DAUMEN_BEUGER_2_AMP
action_amp[0][DAUMEN_ABSPREITZER_1_MOTOR] = DAUMEN_ABSPREITZER_1_AMP


action_phase = np.array([[
    # 0    1    2    3    4    5    6    7
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]])

action_phase[0][ZEIGEFINGER_STRECKER_1_MOTOR] = ZEIGEFINGER_STRECKER_1_PHASE
action_phase[0][ZEIGEFINGER_STRECKER_2_MOTOR] = ZEIGEFINGER_STRECKER_2_PHASE

action_phase[0][ZEIGEFINGER_BEUGER_1_MOTOR] = ZEIGEFINGER_BEUGER_1_PHASE
action_phase[0][ZEIGEFINGER_BEUGER_2_MOTOR] = ZEIGEFINGER_BEUGER_2_PHASE

action_phase[0][DAUMEN_STRECKER_1_MOTOR] = DAUMEN_STRECKER_1_PHASE
action_phase[0][DAUMEN_STRECKER_2_MOTOR] = DAUMEN_STRECKER_2_PHASE

action_phase[0][DAUMEN_BEUGER_2_MOTOR] = DAUMEN_BEUGER_2_PHASE
action_phase[0][DAUMEN_ABSPREITZER_1_MOTOR] = DAUMEN_ABSPREITZER_1_PHASE


# %%

if __name__ == '__main__':
    print(action_amp)
    print(action_off)
    dt_daumen = 0.4
    t1 = [t * 0.005 for t in list(range(1000))]
    t2 = [-np.cos(t - dt_daumen) for t in t1]
    plt.plot(np.cos(t1))
    plt.plot(t2)
    plt.grid()

# %%
