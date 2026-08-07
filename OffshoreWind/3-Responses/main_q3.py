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

from wind import calculateKaimalSpectrum, calculateWindTimeSeries
from common import loadFromJSON, generateRandomPhases
import pylab as plt
import numpy as np
from rotor import F_avg, F_var, F_wind

inputVariables = "inputVariables"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)

# Load the rotor info
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2 * np.pi / 4

# Load the wind info
wind3 = loadFromJSON(fp("wind3.json"))
timeQ3 = loadFromJSON(fp("time.json"))
wind3.update(timeQ3)


# Calculate the time vector
wind3["t"] = np.arange(0., wind3["TDur"], wind3["dt"])

print(f"Computing Kaimal spectrum with V_10={wind3['V_10']} m/s, I={wind3['I']}")

# Compute the time series
wind3 = calculateKaimalSpectrum(wind3)
wind3 = generateRandomPhases(wind3)
wind3 = calculateWindTimeSeries(wind3)

#------------------------------------------
# Calculate time series of wind thrust force
#------------------------------------------
# Initialize force arrays for time series
wind3['F_avg'] = np.zeros_like(wind3['t'])
wind3['F_var'] = np.zeros_like(wind3['t'])
wind3['F_total'] = np.zeros_like(wind3['t'])

# Calculate forces for each time step
for i_, t_ in enumerate(wind3['t']):
    wind3['F_avg'][i_] = F_avg(iea22mw, wind3['V_10'])
    wind3['F_var'][i_] = F_var(iea22mw, wind3['V_hub'][i_])
    wind3['F_total'][i_] = F_wind(iea22mw, wind3['V_10'],wind3['V_hub'][i_])

# Plot wind speed time series
plt.figure(figsize=(12, 4))
plt.plot(wind3['t'], wind3['V_hub'])
plt.xlabel('Time [s]')
plt.ylabel('Wind Speed [m/s]')
plt.title('Wind Speed Time Series at Hub Height')
plt.grid(True)
plt.show()

# Plot variable wind force time series
plt.figure(figsize=(12, 4))
plt.plot(wind3['t'], wind3['F_total'])
plt.xlabel('Time [s]')
plt.ylabel('Total Force [N]')
plt.title('Total Wind Thrust Force Time Series')
plt.grid(True)
plt.show()

'''
# Plot total wind force time series
plt.figure(figsize=(12, 4))
plt.plot(wind3['t'], wind3['F_total'], label='Total Force')
plt.plot(wind3['t'], wind3['F_avg'], label='Average Force', linestyle='--')
plt.plot(wind3['t'], wind3['F_var'], label='Variable Force', alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Wind Thrust Force Components')
plt.legend()
plt.grid(True)
plt.show()
'''

#------------------------------------------
# Statistics of the wind speed time series
#------------------------------------------
print(f"Wind speed:")
print(f"  Mean: {np.mean(wind3['V_hub']):.2f} m/s")
print(f"  Std: {np.std(wind3['V_hub']):.2f} m/s")
print(f"  Min: {np.min(wind3['V_hub']):.2f} m/s")
print(f"  Max: {np.max(wind3['V_hub']):.2f} m/s")

#------------------------------------------
# Statistics of the wind force time series
#------------------------------------------
print(f"\nTotal Wind force:")
print(f"  Mean: {np.mean(wind3['F_total'])/1e6:.3f} MN")
print(f"  Std: {np.std(wind3['F_total'])/1e6:.3f} MN")
print(f"  Min: {np.min(wind3['F_total'])/1e6:.3f} MN")
print(f"  Max: {np.max(wind3['F_total'])/1e6:.3f} MN")

