# %%
import numpy as np
import matplotlib.pyplot as plt
import json
import transformation_functions as tf


if __name__ == "__main__":
    path = 'Data/test_01_31/2023_01_31_18_12_48.json'
    f = open(path)
    data = json.load(f)
    f.close()

    lalbes = ['Thumb Spreader', 'Thumb flexor', 'Flexor 2 pointer finger', 'Flexor 1 pointer finger', \
              'extensor 1 pointer finger', 'extensor 2 thumb', 'extensor 2 pointer finger', 'extensor 1 thumb']

    time = data['time']
    forces = []
    for i in range(8):
        forces.append(data['observation']['analogs'][i]['dforce'])
        plt.plot(time,forces[i], label=lalbes[i])
    plt.legend(loc='upper right')
    plt.show()
# %% Berechnung der Markerpositionen im CT