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
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','functions', 'python'))
sys.path.append(helpers_path)

import pylab as plt
from common import loadConstants, loadFromJSON, downsample, saveToJSON    
import os.path
import numpy as np
from loads import calculateStaticWindLoads, calculateStaticWaveLoads, calculateDynamicLoads
from monopile import computeElementwiseQuantities

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

# Load the structural motions
q = loadFromJSON(ss("q.json"))
alphaDotDot = np.gradient(q["alphaDot"], q["t"])
q["alphaDotDot"] = alphaDotDot

# Load the wind and waves
wind = loadFromJSON(ss("wind4.json"))
waves = loadFromJSON(ss("waves4.json"))

# Load the monopile and the rotor
iea22mw = loadFromJSON(fp("iea22mw.json"))

monopileDict = loadFromJSON(fp("monopile.json"))

# FIXME Assignment 3 Q1.7: look inside calculate****Loads and fix

# Calculate the static wind loads
windDownsampled = downsample(wind, dropEvery=2, listOfFields=["t", "V_hub"])
windLoads = calculateStaticWindLoads(windDownsampled, iea22mw, 
                                monopileDict, q)

# Calculate the static wave loads
wavesDownsampled = downsample(waves, dropEvery=2, listOfFields=["t", "u", "ut"])
waveLoads = calculateStaticWaveLoads(wavesDownsampled, monopileDict, q)

# Calculate the dynamic loads
monopileDict = computeElementwiseQuantities(monopileDict)
dynamicLoads = calculateDynamicLoads(monopileDict, q)

saveToJSON(dynamicLoads, "savedStates/dynLoads.json")

# Figure
plt.figure()
plt.plot(dynamicLoads["t"], dynamicLoads["M"], label="Dynamic")
plt.plot(windLoads["t"], windLoads["M"], label="Wind")
plt.plot(waveLoads["t"], waveLoads["M"], label="Waves")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Moment at mudline [m]") 
plt.title("Moments at Seabed in function of time")
plt.legend()
plt.show()

# Print the results
print(f"\nWind Loads:")
print(f"  Mean: {np.mean(windLoads['M']):.3e} Nm")
print(f"  Std: {np.std(windLoads['M']):.3e} Nm")
print(f"  Min: {np.min(windLoads['M']):.3e} Nm")
print(f"  Max: {np.max(windLoads['M']):.3e} Nm")

print(f"\nWave Loads:")
print(f"  Mean: {np.mean(waveLoads['M']):.3e} Nm")
print(f"  Std: {np.std(waveLoads['M']):.3e} Nm")
print(f"  Min: {np.min(waveLoads['M']):.3e} Nm")
print(f"  Max: {np.max(waveLoads['M']):.3e} Nm")

mask = dynamicLoads['t'] >= 60
M_mask = dynamicLoads['M'][mask]

print(f"\nDynamic Loads:")
print(f"  Mean: {np.mean(M_mask):.3e} Nm")
print(f"  Std: {np.std(M_mask):.3e} Nm")
print(f"  Min: {np.min(M_mask):.3e} Nm")
print(f"  Max: {np.max(M_mask):.3e} Nm")

print(np.mean(alphaDotDot))
print(np.max(alphaDotDot))