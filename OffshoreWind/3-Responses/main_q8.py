'''
Filename: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\reports\\Report3\\2024\\solution\\python\\classicalSolution\\main_q8.py
Path: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\reports\\Report3\\2024\\solution\\python\\classicalSolution
Created Date: Thursday, October 10th 2024, 10:27:58 pm
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''
import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeries, calculateKinematics, calculateFreeSurfaceElevationTimeSeriesFFT, calculateKinematicsFFT
from wind import calculateWindTimeSeries, calculateWindTimeSeriesFFT
from common import loadFromJSON, Timer
import pylab as plt
import os.path
import numpy as np

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

waves = loadFromJSON(ss("waves4.json"))
timeInfo = loadFromJSON(fp("time.json"))
waves.update(timeInfo)

# Calculate the time vector
waves["t"] = np.arange(0., waves["TDur"], waves["dt"])

# Calculate the jonswap spectrum
waves = calculateJONSWAPSpectrum(waves)


with Timer("slow waves"):
    waves = calculateFreeSurfaceElevationTimeSeries(waves)
    waves = calculateKinematics(waves)

# Now with FFT
# FIXME Assignment 3 Q1.8: implement the FFT functions in waves.py
with Timer("fast waves"):
    wavesFast = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    wavesFast = calculateKinematicsFFT(wavesFast)
   
# Load the wind info
wind = loadFromJSON(ss("wind4.json"))
wind.update(timeInfo)

randomWind = 11
with Timer("Slow wind"):
    wind = calculateWindTimeSeries(wind)
    
# FIXME Assignment 3 Q1.8: implement the FFT functions in wind.py
with Timer("Fast wind"):
    windFast = calculateWindTimeSeriesFFT(wind)    



fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axs[0].plot(waves["eta"] - wavesFast["eta"], '.')
axs[0].set_ylabel(r"$\eta$ - $\eta_{FFT}$")
axs[0].set_title("Difference: Free Surface Elevation")
axs[0].grid(True)

axs[1].plot(waves["u"][:,-1] - wavesFast["u"][:,-1], '.')
axs[1].set_ylabel(r"$u(end)$ - $u_{FFT}(end)$")
axs[1].set_title("Difference: Horizontal Velocity at Top Node")
axs[1].grid(True)

axs[2].plot(waves["ut"][:,-1] - wavesFast["ut"][:,-1], '.')
axs[2].set_ylabel(r"$u_t(end)$ - $u_{t_{FFT}}(end)$")
axs[2].set_title("Difference: Horizontal Acceleration at Top Node")
axs[2].grid(True)

axs[3].plot(wind["V_hub"] - windFast["V_hub"], '.')
axs[3].set_ylabel(r"$V_{hub}$ - $V_{hub_{FFT}}$")
axs[3].set_title("Difference: Wind at Hub Height")
axs[3].set_xlabel("Time index")
axs[3].grid(True)

plt.tight_layout()
plt.show()