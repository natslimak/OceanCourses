'''
Filename: c:\\Users\\fabpi\\OneDoc\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Lectures\\classical\\main_q2.py
Path: c:\\Users\\fabpi\\OneDoc\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Lectures\\classical
Created Date: Monday, September 30th 2024, 2:22:21 pm
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''
import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ,'..','functions', 'python'))
sys.path.append(helpers_path)

from rotor import *
from wind import calculateWindTimeSeries
from common import loadFromJSON
import numpy as np
import pylab as plt

inputVariables = "inputVariables"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)

iea22mw = loadFromJSON(fp("iea22mw.json"))
wind3 = loadFromJSON(fp("wind3.json"))
time= loadFromJSON(fp("time.json"))

thrust = dict()
thrust["V"] = np.arange(0., 25.)
thrust["T"] = np.zeros_like(thrust["V"])

for i_, V_ in enumerate(thrust["V"]):
    thrust["T"][i_] = F_avg(iea22mw, V_)

plt.plot(thrust["V"], thrust["T"])
plt.xlabel('Wind speed V [m/s]')
plt.ylabel('Thrust force T [N]')
plt.grid()
plt.title('Trust force in function of the wind speed')
plt.show()

"""
Force = dict()
Force["V"] = np.arange(3., 25.)
Force["F_avg"] = np.zeros_like(Force["V"])
Force["F_var"] = np.zeros_like(Force["V"])


V_hub=calculateWindTimeSeries(wind3)


# FIXME: Assignment 3 Q1.2: fix the rotor aerodynamic model, see inside in rotor.py
for i_, V_ in enumerate(Force["V"]):
    Force["F_avg"][i_] = F_avg(iea22mw, wind3['V_10'])
    Force['F_var'][i_] = F_var(iea22mw, V_hub['V_hub'])
    Force['F'][i_] =Force['F_var'][i_]+Force["F_avg"][i_]
      

plt.plot(Force['V'],Force['F'])
plt.show()
"""