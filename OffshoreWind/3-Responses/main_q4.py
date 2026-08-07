'''
Filename: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Lectures\\classical\\main_q1.py
Path: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Lectures\\classical
Created Date: Monday, September 30th 2024, 12:04:08 pm
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''
import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','functions', 'python'))
sys.path.append(helpers_path)

from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeries, calculateKinematics
from wind import calculateKaimalSpectrum, calculateWindTimeSeries
from common import loadFromJSON, saveToJSON, generateRandomPhases
from monopile import forceIntegrate
from rotor import F_wind
import pylab as plt
import os.path
import numpy as np

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

# Load the wind info
# FIXME Assignment 3 Q1.4: check correctness of json input dicts
wind4 = loadFromJSON(fp("wind4.json"))
time4 = loadFromJSON(fp("time.json"))
wind4.update(time4)

# Load the waves stuff
waves4 = loadFromJSON(fp("wave1.json"))
waves4.update(time4)

wind4["t"] = np.arange(0., wind4["TDur"], wind4["dt"])
waves4["t"] = np.arange(0., waves4["TDur"], waves4["dt"])

# Compute the time series
wind4 = calculateKaimalSpectrum(wind4)
wind4 = generateRandomPhases(wind4, seed=1)
wind4 = calculateWindTimeSeries(wind4)

# Compute waves
waves4 = calculateJONSWAPSpectrum(waves4)
waves4 = generateRandomPhases(waves4, seed=2)
waves4 = calculateFreeSurfaceElevationTimeSeries(waves4)
waves4 = calculateKinematics(waves4)
h = waves4["h"]

# Load the monopile
monopileDict = loadFromJSON(fp("monopile.json"))

# Wave force
waveForce = dict()
waveForce["t"] = waves4["t"]
waveForce["F"], waveForce["M"] = np.zeros_like(waves4["t"]), np.zeros_like(waves4["t"])

for i_, t_ in enumerate(waves4["t"]):
	# FIXME Assignment 3 Q1.4: call the forceIntegrate function from monopile.py to get the wave loads
    waveForce["F"][i_], waveForce["M"][i_]  = forceIntegrate(monopileDict, waves4["u"][i_,:], waves4["ut"][i_,:],
        waves4["z"], 0., waves4["z"][0]) # Moment at seabed

# Wind force

iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi/4

windForce = dict()
windForce["t"] = wind4["t"]
windForce["F"], windForce["M"] = np.zeros_like(wind4["t"]), np.zeros_like(wind4["t"])

for i_, t_ in enumerate(wind4["t"]):
	# FIXME Assignment 3 Q1.4: call the F_wind from wind.py to get the total wind force
    windForce["F"][i_] = F_wind(iea22mw, wind4["V_10"], wind4["V_hub"][i_])    
    windForce["M"][i_] = windForce["F"][i_]*(monopileDict["zBeamNodal"][-1]+h)

# ----------------------------------------------
# Calculating total force wind and waves
# ----------------------------------------------
totalForce = dict()
totalForce["t"] = wind4["t"]
totalForce["F"], totalForce["M"] = np.zeros_like(wind4["t"]), np.zeros_like(wind4["t"])

for i_ in range(len(wind4["t"])):
    totalForce["F"][i_] = windForce["F"][i_] + waveForce["F"][i_]
    totalForce["M"][i_] = windForce["M"][i_] + waveForce["M"][i_]

'''    
plt.figure()
plt.plot(waves4["t"], waves4["eta"])

plt.figure()
plt.plot(waveForce["t"], waveForce["F"])
'''

# Save for later use
os.makedirs(savedStates, exist_ok=True)
saveToJSON(waves4, ss("waves4.json"))
saveToJSON(wind4, ss("wind4.json" ))

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(windForce["t"], windForce["M"], label='Wind Load', color='blue')
plt.plot(waveForce["t"], waveForce["M"], label='Wave Load', color='green')
plt.plot(totalForce["t"], totalForce["M"], label='Total Force', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Overturning Moment [Nm]')
plt.title('Overturning Moment at Seabed - Combined Wind and Wave Loading')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------
# Statistics
# ----------------------------------------------
print(f"\nTotal Wind Moment:")
print(f"  Mean: {np.mean(windForce['M'])/1e6:.3f} MNm")
print(f"  Std: {np.std(windForce['M'])/1e6:.3f} MNm")
print(f"  Min: {np.min(windForce['M'])/1e6:.3f} MNm")
print(f"  Max: {np.max(windForce['M'])/1e6:.3f} MNm")

print(f"\nTotal Wave Moment:")
print(f"  Mean: {np.mean(waveForce['M'])/1e6:.3f} MNm")
print(f"  Std: {np.std(waveForce['M'])/1e6:.3f} MNm")
print(f"  Min: {np.min(waveForce['M'])/1e6:.3f} MNm")
print(f"  Max: {np.max(waveForce['M'])/1e6:.3f} MNm")

print(f"\nTotal Combined Moment:")
print(f"  Mean: {np.mean(totalForce['M'])/1e6:.3f} MNm")
print(f"  Std: {np.std(totalForce['M'])/1e6:.3f} MNm")
print(f"  Min: {np.min(totalForce['M'])/1e6:.3f} MNm")
print(f"  Max: {np.max(totalForce['M'])/1e6:.3f} MNm")

print(np.mean(totalForce["F"]))