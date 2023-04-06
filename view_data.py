# %%

#import numpy as np
import json
import matplotlib.pyplot as plt
#from dataclasses import dataclass
#from ipywidgets import widgets
import os
#from pyquaternion import Quaternion

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
# Opening JSON file
f = open('Data/test_01_30/2023_01_31_00_47_54.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
f.close()

print('data loaded')
print(data.keys())
print('len of time:',len(data['time']))
print('len of observation:', len(data['observation']))
print(data['observation'].keys())
print(data['observation']['rigid_bodies'])
print('len of motors:', len(data['observation']['motors']))
print('len of action:',len(data['action']))

print(data['time'][0:10])
print(data['action'][10])
#print(len(data['observation']['motors'][0][0]))

'''Prin the actions.'''
x = data['time'][:100]
y = data['action'][:100]
plt.xlabel("time [ms]")
plt.ylabel("action value")
plt.title("Actions")
for i in range(8):
    plt.plot(x,[pt[0][i] for pt in y],label = 'action %s'%i)
plt.legend()
plt.show()

'''Rigid Bodies Plot Raw'''
for i in range(len(data['observation']['rigid_bodies'])):
    start = 0
    stop = -1
    plt.plot(data['time'][start:stop],data['observation']['rigid_bodies'][i]['x'][start:stop], label = 'x')
    plt.plot(data['time'][start:stop],data['observation']['rigid_bodies'][i]['y'][start:stop], label = 'y')
    plt.plot(data['time'][start:stop],data['observation']['rigid_bodies'][i]['z'][start:stop], label = 'z')
    plt.title("Rigid Body %s"%i)
    plt.legend()
    plt.show()

# %%
