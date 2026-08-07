'''
Filename: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Code\\classicalSolution\\Stud_\\main_q9.py
Path: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Code\\classicalSolution\\Stud_
Created Date: Friday, October 18th 2024, 12:13:12 pm
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''
import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..','functions', 'python'))
sys.path.append(helpers_path)

import os
import pandas as pd
from common import loadFromJSON, Timer
import numpy as np
from runner import runEnvironmentalCondition
import rainflow
import matplotlib.pyplot as plt 

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

# Load the rotor and the monopile which do not change
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["VCutIn"] = 3.0; iea22mw["VCutOut"] = 25.0
monopile = loadFromJSON(fp("monopile.json"))

# Time information
timeInfo = dict()
timeInfo["TDur"] = 3660.
timeInfo["TTrans"] = 60.
timeInfo["dt"] = 0.05
timeInfo["fHighCut"] = 0.5

# Fatigue parameters
mFatigue = 4.
n_eq = 10.**7

# Time factor
# FIXME Assignment 3 Q11: Calculate TLife and Tsim
TLife = 25*365*24*3600 #25 years
TSim = 3600 #1 hour

# Rescale wind speed by taking into account shear factor 
# FIXME Assignment 3 Q11 include the correct scaleWind parameter using a shear factor of 1/7 and factor 2 on hub height
scaleWind = 1

# Load the environmental conditions
wind = loadFromJSON(fp("wind4.json"))    
waves = loadFromJSON(fp("wave1.json"))
wind["V_10"] = wind["V_10"]*scaleWind
wind["randomSeed"], waves["randomSeed"] = 1, 2

# Run the loads calculation
Output, response= runEnvironmentalCondition(wind,
                    waves,
                    iea22mw,
                    monopile,
                    timeInfo, rotor_state="on")

outputLoads=Output["total"]
                        
# Remove transient
Filter = outputLoads["t"] >= timeInfo["TTrans"]
outputLoads["t"] = outputLoads["t"][Filter]
outputLoads["F"] = outputLoads["F"][Filter]
outputLoads["M"] = outputLoads["M"][Filter]

# Rainflow count


# FIXME Assignment 3 Q11 do the rainflow counting of fatigue here (see slides for example)
rainflowCount = np.array(rainflow.count_cycles(outputLoads['M'])) # apply the rainflow.count_cycles routine
amplitude = rainflowCount[:,0]/2 # transform range into amplitude
cycles = rainflowCount[:,1]
cyclesUpscaled = cycles*TLife/TSim

# FIXME Assignment 3 Q11 calculate the equivalent moment here 
M_eq = (np.sum(cyclesUpscaled*amplitude**mFatigue)/n_eq)**(1/mFatigue)

# Plot histogram of stress amplitude
plt.figure(figsize=(10,6))
plt.title('Stress Amplitude Histogram')
plt.hist(amplitude,bins=30, edgecolor = 'black')
plt.xlabel(r'Moment amplitude $M$ [Nm]')
plt.ylabel(r'Number of cycles $n$')
plt.show()

print("M_eq=",M_eq)