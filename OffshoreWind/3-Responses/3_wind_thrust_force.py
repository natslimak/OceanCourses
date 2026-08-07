""" Calculate the wind thrust force time series for a given wind speed time series. """

import os
import sys 
import pylab as plt
import numpy as np

# Add the functions folder to the path 
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON, generateRandomPhases
from rotor import F_avg, F_var, F_wind
from wind import calculateKaimalSpectrum, calculateWindTimeSeries


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)

# Load the rotor, wind and time info
iea22mw = loadFromJSON(get_input_file("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2 * np.pi / 4
wind_data = loadFromJSON(get_input_file("wind.json"))
time_data = loadFromJSON(get_input_file("time.json"))
wind_data.update(time_data)
wind_data["t"] = np.arange(0., wind_data["TDur"], wind_data["dt"])

# Compute the Kaimal spectrum and generate a wind speed time series at hub height
wind_data = calculateKaimalSpectrum(wind_data)
randomSeedWind = 2
wind_data = generateRandomPhases(wind_data, seed=randomSeedWind)
wind_data = calculateWindTimeSeries(wind_data)

print(f"Computing Kaimal spectrum with V_10={wind_data['V_10']} m/s, I={wind_data['I']}")


# ============================================================================
# Calculate time series of wind thrust force
# ============================================================================
# Initialize force arrays for time series
wind_data['F_avg'] = np.zeros_like(wind_data['t'])
wind_data['F_var'] = np.zeros_like(wind_data['t'])
wind_data['F_total'] = np.zeros_like(wind_data['t'])

# Calculate forces for each time step
for i_, t_ in enumerate(wind_data['t']):
    wind_data['F_avg'][i_] = F_avg(iea22mw, wind_data['V_10'])
    wind_data['F_var'][i_] = F_var(iea22mw, wind_data['V_hub'][i_])
    wind_data['F_total'][i_] = F_wind(iea22mw, wind_data['V_10'],wind_data['V_hub'][i_])


# ============================================================================
# Plots: Wind speed and thrust force time series
# ============================================================================

# Plot wind speed time series
plt.figure(figsize=(12, 4))
plt.plot(wind_data['t'], wind_data['V_hub'])
plt.xlabel('Time [s]')
plt.ylabel('Wind Speed [m/s]')
plt.title('Wind Speed Time Series at Hub Height')
plt.grid(True)
plt.show()

# Plot variable wind force time series
plt.figure(figsize=(12, 4))
plt.plot(wind_data['t'], wind_data['F_total'])
plt.xlabel('Time [s]')
plt.ylabel('Total Force [N]')
plt.title('Total Wind Thrust Force Time Series')
plt.grid(True)
plt.show()


# ============================================================================
# Statistics of the wind speed time series and the total wind force time series
# ============================================================================
print(f"Wind speed:")
print(f"  Mean: {np.mean(wind_data['V_hub']):.2f} m/s")
print(f"  Std: {np.std(wind_data['V_hub']):.2f} m/s")
print(f"  Min: {np.min(wind_data['V_hub']):.2f} m/s")
print(f"  Max: {np.max(wind_data['V_hub']):.2f} m/s")

print(f"\nTotal Wind force:")
print(f"  Mean: {np.mean(wind_data['F_total'])/1e6:.3f} MN")
print(f"  Std: {np.std(wind_data['F_total'])/1e6:.3f} MN")
print(f"  Min: {np.min(wind_data['F_total'])/1e6:.3f} MN")
print(f"  Max: {np.max(wind_data['F_total'])/1e6:.3f} MN")

