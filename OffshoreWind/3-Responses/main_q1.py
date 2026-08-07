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
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from waves import *
from common import *
from monopile import forceIntegrate
import pylab as plt
import os.path
import numpy as np

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)

# Path function for assignment1 files (for wave5.json)
fp_a1 = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', 'assignment1', 'python', 'inputVariables', x)

# Question 1
# FIXME: input the correct parameters inside wave1.json OK
wavesQ1 = loadFromJSON(fp("wave1.json"))
    
# Load the time discretization info
timeQ1 = loadFromJSON(fp("time.json"))
wavesQ1.update(timeQ1)

# Calculate the time vector
wavesQ1["t"] = np.arange(0., wavesQ1["TDur"], wavesQ1["dt"])

# Calculate the jonswap spectrum
wavesQ1 = calculateJONSWAPSpectrum(wavesQ1)

randomSeedWaves = 1
wavesQ1 = generateRandomPhases(wavesQ1, seed=randomSeedWaves)

wavesQ1 = calculateFreeSurfaceElevationTimeSeries(wavesQ1)
wavesQ1 = calculateKinematics(wavesQ1)

monopileDict = loadFromJSON(fp("monopile.json"))

forceQ1 = dict()
forceQ1["t"] = wavesQ1["t"]
forceQ1["F"], forceQ1["M"] = np.zeros_like(wavesQ1["t"]), np.zeros_like(wavesQ1["t"])

for i_, t_ in enumerate(wavesQ1["t"]):
    forceQ1["F"][i_], forceQ1["M"][i_]  = forceIntegrate(monopileDict, wavesQ1["u"][i_,:], wavesQ1["ut"][i_,:],
        wavesQ1["z"], 0., wavesQ1["z"][0]) # Moment at seabed
    
'''
plt.figure()
plt.plot(wavesQ1["t"], wavesQ1["eta"])

plt.figure()
plt.plot(forceQ1["t"], forceQ1["M"])
plt.show()
'''

# -------------------------------------------------
# Calulating overturning moment for Hs = stddev*4
#--------------------------------------------------

# After calculating the wave spectrum and time series, add validation:
# Check if 4*sigma_eta = Hs
sigma_eta = np.std(wavesQ1["eta"])
expected_sigma = wavesQ1["Hs"] / 4.0
scaling_factor = expected_sigma / sigma_eta

print(f"Current σ_η: {sigma_eta:.3f}")
print(f"Expected σ_η (Hs/4): {expected_sigma:.3f}")
print(f"Scaling factor needed: {scaling_factor:.3f}")

# Apply scaling if needed
if abs(scaling_factor - 1.0) > 0.01:  # If scaling is significantly needed   -> could have been also directly scaled the AMPLITUDE spectrum
     wavesQ1["eta"] *= scaling_factor
     # Also scale velocities and accelerations accordingly
     wavesQ1["u"] *= scaling_factor
     wavesQ1["ut"] *= scaling_factor
     print(f"Applied scaling factor: {scaling_factor:.3f}")

# Verify the scaling
sigma_eta_corrected = np.std(wavesQ1["eta"])
print(f"Corrected σ_η: {sigma_eta_corrected:.3f}")
print(f"Ratio 4σ_η/Hs: {4*sigma_eta_corrected/wavesQ1['Hs']:.3f}")


# -------------------------------------------------
# Piece of code from assignment1/python/main_q2.py
#--------------------------------------------------
'''
wavesQ5 = loadFromJSON(fp_a1("wave5.json"))
timeQ5 = loadFromJSON(fp("time.json"))
wavesQ5.update(timeQ5)
wavesQ5["t"] = np.arange(0., wavesQ5["TDur"], wavesQ5["dt"])


wavesQ5 = calculateJONSWAPSpectrum(wavesQ5)
wavesQ5 = generateRandomPhases(wavesQ5, seed=randomSeedWaves)
wavesQ5 = calculateKinematics(wavesQ5)
wavesQ5 = calculateFreeSurfaceElevationTimeSeries(wavesQ5)

forceQ5 = dict()
forceQ5["t"] = wavesQ5["t"]
forceQ5["F"], forceQ5["M"] = np.zeros_like(wavesQ5["t"]), np.zeros_like(wavesQ5["t"])

for i_, t_ in enumerate(wavesQ5["t"]):
    forceQ5["F"][i_], forceQ5["M"][i_]  = forceIntegrate(monopileDict, wavesQ5["u"][i_,:], np.zeros_like(wavesQ5["ut"][i_,:]),
        wavesQ5["z"], 0.)
'''
# -------------------------------------------------
# Plot
#--------------------------------------------------
'''
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# First subplot - Q1 results
ax1.plot(forceQ1["t"], forceQ1["M"])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Overturning Moment [Nm]')
ax1.set_title('Overturning Moment at Seabed - Q1')
ax1.grid(True)

# Second subplot - Q5 results
ax2.plot(wavesQ5["t"], forceQ5["M"])
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Overturning Moment [Nm]')
ax2.set_title('Overturning Moment at Seabed - Q5')
ax2.grid(True)

plt.tight_layout()
plt.show()
'''

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top subplot - Overturning Moment
ax1.plot(forceQ1["t"], forceQ1["M"])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Overturning Moment [Nm]')
ax1.set_title('Overturning Moment at Mudline')
ax1.grid(True)

# Bottom subplot - Wave Elevation
ax2.plot(wavesQ1["t"], wavesQ1["eta"])
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Wave Elevation [m]')
ax2.set_title('Wave Elevation at Sea Surface')
ax2.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------
# Statistical Analysis
#--------------------------------------------------
print(f"Wave elevation statistics:")
print(f"  Mean: {np.mean(wavesQ1["eta"]):.4f} m")
print(f"  Std: {np.std(wavesQ1["eta"]):.4f} m")
print(f"  Min: {np.min(wavesQ1["eta"]):.4f} m")
print(f"  Max: {np.max(wavesQ1["eta"]):.4f} m")


print(f"Overturning moment statistics:")
print(f"  Mean: {np.mean(forceQ1["M"])/1e6:.4f} MNm")
print(f"  Std: {np.std(forceQ1["M"])/1e6:.4f} MNm")
print(f"  Min: {np.min(forceQ1["M"])/1e6:.4f} MNm")
print(f"  Max: {np.max(forceQ1["M"])/1e6:.4f} MNm")

# NOTE!!!!!!

# At seabed, z = -34 m
# for i_, t_ in enumerate(wavesQ1["t"]):
#    forceQ1["F"][i_], forceQ1["M"][i_]  = forceIntegrate(monopileDict, wavesQ1["u"][i_,:], wavesQ1["ut"][i_,:], wavesQ1["z"], 0., wavesQ1["z"][0])

# At mudline
# for i_, t_ in enumerate(wavesQ1["t"]):
#    forceQ1["F"][i_], forceQ1["M"][i_]  = forceIntegrate(monopileDict, wavesQ1["u"][i_,:], wavesQ1["ut"][i_,:], wavesQ1["z"], 0.)

