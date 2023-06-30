import matplotlib.pyplot as plt
import numpy as np

def plot_analogs_angles(angles=[], flexor=[], extensor=[], time=[], step_size=5000, legend=[], title='', save_plots=False):
    pos_x = np.arange(max(time), step=step_size)  # type: ignore
    x = [int(i / 1000) for i in pos_x]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(time, angles)
    ax2.plot(time, flexor)
    ax3.plot(time, extensor)
    ax1.set_ylabel('angle [°]')
    ax2.set_ylabel('force [N]')
    ax3.set_ylabel('force [N]')
    ax3.set_xlabel('time [sec]')
    ax1.set_title(title)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    plt.xticks(pos_x, x)
    plt.show()
    plt.close()

# Example usage:
angles = [10, 20, 30, 40, 50]
flexor = [5, 10, 15, 20, 25]
extensor = [25, 20, 15, 10, 5]
time = [1, 2, 3, 4, 5]
title = 'Analog Angles'
plot_analogs_angles(angles, flexor, extensor, time, title=title)
