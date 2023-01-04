# %%

#import numpy as np
import json
#import matplotlib.pyplot as plt
#from dataclasses import dataclass
#from ipywidgets import widgets
#import os
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
f = open('data/test_januar/16_21_19.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
print('data loaded')
print(data.keys())
print('len of time:',len(data['time']))
print('len of observation:', len(data['observation']))
print(data['observation'].keys())
print('len of motors:', len(data['observation']['motors']))
print('len of action:',len(data['action']))

print(data['time'][0:10])
print(data['action'][10])
#print(len(data['observation']['motors'][0][0]))

f.close()