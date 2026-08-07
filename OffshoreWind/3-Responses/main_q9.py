'''
Filename: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\reports\\Report3\\2024\\solution\\python\\classicalSolution\\main_q7.py
Path: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\reports\\Report3\\2024\\solution\\python\\classicalSolution
Created Date: Monday, October 7th 2024, 11:52:58 am
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''

import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

import pylab as plt
from common import loadConstants, loadFromJSON, downsample, saveToJSON    
import os.path
import numpy as np
from loads import calculateStaticWindLoads, calculateStaticWaveLoads, calculateDynamicLoads
from monopile import computeElementwiseQuantities, forceIntegrate
from plotting import makeplotsMonopile
from rotor import F_wind
from runner import runEnvironmentalCondition

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

# FIXME:
# 1) create the appropriate dictionaries
# Load the dictionaries
monopileDict = loadFromJSON(fp("monopile.json"))
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi*0.25
waves = loadFromJSON(ss("waves4.json")) # Does not change comaring to q8
wind = loadFromJSON(ss("wind4.json"))   # Does not change comparing to q8
timeInfo = loadFromJSON(fp("time.json"))
q = loadFromJSON(ss("q.json"))

wind["I"] = 0 # Set turbulence intensity to zero
wind["V_hub"][:] = wind["V_10"]

envDict, response = runEnvironmentalCondition(wind, waves, iea22mw, monopileDict, timeInfo, rotor_state="on")

totalF = envDict["total"]["F"]
totalM = envDict["total"]["M"]
totaltt = envDict["total"]["t"]
totalLoads = {"t": totaltt, "F": totalF, "M": totalM}

# 2) re-run similar code as Q6 and Q7 to compute the loads and save them inside totalLoads, which has keys "t", "F" and "M". Response is a dict with "t", "alpha", "alphaDot", "alphaDotDot".

# 3) plot the results using the makeplotsMonopile function
                       
# Assignment 3 Q9
# -----------------------------                       
ax = makeplotsMonopile(wind, waves, totalLoads, response, timeInfo, "r", alpha=0.5)


# Assignment 3 Q10
# -----------------------------
monopileDict = loadFromJSON(fp("monopile.json"))
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi*0.25
waves = loadFromJSON(ss("waves4.json")) # Does not change comaring to q8
wind = loadFromJSON(ss("wind4.json"))   # Does not change comparing to q8
timeInfo = loadFromJSON(fp("time.json"))
q = loadFromJSON(ss("q.json"))

wind["I"] = 0
wind["V_hub"][:] = wind["V_10"]


envDict, response = runEnvironmentalCondition(wind, waves, iea22mw, monopileDict, timeInfo, rotor_state="off")

totalF = envDict["total"]["F"]
totalM = envDict["total"]["M"]
totaltt = envDict["total"]["t"]
totalLoadsRotorOff = {"t": totaltt, "F": totalF, "M": totalM}
responseRotorOff = response

print(wind)

# show vertical marker at t = 60 s on every subplot (do not remove data)
for a in np.array(ax).flatten():
    a.axvline(60.0, color='k', linestyle='--', linewidth=1, zorder=2)

# add a "60 s" label at the top of the top-left subplot
ymax = ax[0,0].get_ylim()[1]
ax[0,0].text(60.0, ymax*0.98, '60 s', color='k', ha='left', va='top',
             fontsize=9, backgroundcolor='white')


makeplotsMonopile(wind, waves, totalLoadsRotorOff, responseRotorOff, timeInfo, "b", ax=ax, alpha=0.5)

ax[0,0].legend(["Rotor on", "Rotor off"])
plt.show()