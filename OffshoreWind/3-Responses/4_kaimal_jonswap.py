""" Wind and Wave Loading on a Monopile Offshore Wind Turbine"""

import os
import sys 
import pylab as plt
import numpy as np

# Add the functions folder to the path 
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeries, calculateKinematics
from wind import calculateKaimalSpectrum, calculateWindTimeSeries
from common import loadFromJSON, saveToJSON, generateRandomPhases
from monopile import forceIntegrate
from rotor import F_wind

# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', '..', output_dir, fname)

# Load the monopile, wind and wave data, as well as the time properties
monopile_props = loadFromJSON(get_input_file("monopile.json"))
wind_data = loadFromJSON(get_input_file("wind.json"))
time_data = loadFromJSON(get_input_file("time.json"))
wave_data = loadFromJSON(get_input_file("wave_irregular.json"))

wind_data.update(time_data)
wave_data.update(time_data)

wind_data["t"] = np.arange(0., wind_data["TDur"], wind_data["dt"])
wave_data["t"] = np.arange(0., wave_data["TDur"], wave_data["dt"])

# Compute the Kaimal spectrum and generate a wind speed time series at hub height
randomSeedWind = 1
wind_data = calculateKaimalSpectrum(wind_data)
wind_data = generateRandomPhases(wind_data, seed=randomSeedWind)
wind_data = calculateWindTimeSeries(wind_data)

# Compute the JONSWAP spectrum and generate a wave time series
randomSeedWave = 2
wave_data = calculateJONSWAPSpectrum(wave_data)
wave_data = generateRandomPhases(wave_data, seed=randomSeedWave)
wave_data = calculateFreeSurfaceElevationTimeSeries(wave_data)
wave_data = calculateKinematics(wave_data)
h = wave_data["h"]


# ============================================================================
# Get the wind thrust force and wave force time series
# ============================================================================

# WAVE FORCE
waveForce = dict()
waveForce["t"] = wave_data["t"]
waveForce["F"], waveForce["M"] = np.zeros_like(wave_data["t"]), np.zeros_like(wave_data["t"])

for i_, t_ in enumerate(wave_data["t"]):
    waveForce["F"][i_], waveForce["M"][i_]  = forceIntegrate(monopile_props, wave_data["u"][i_,:], wave_data["ut"][i_,:],
        wave_data["z"], 0., wave_data["z"][0]) # Moment at seabed


# WIND FORCE
iea22mw = loadFromJSON(get_input_file("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi/4

windForce = dict()
windForce["t"] = wind_data["t"]
windForce["F"], windForce["M"] = np.zeros_like(wind_data["t"]), np.zeros_like(wind_data["t"])

for i_, t_ in enumerate(wind_data["t"]):
    windForce["F"][i_] = F_wind(iea22mw, wind_data["V_10"], wind_data["V_hub"][i_])    
    windForce["M"][i_] = windForce["F"][i_]*(monopile_props["zBeamNodal"][-1]+h)


# ============================================================================
# Calculating total force wind and waves
# ============================================================================
# Ensure wind and wave time series have the same length before combining
nt_wind = len(wind_data["t"])
nt_wave = len(wave_data["t"])
nt = min(nt_wind, nt_wave)
if nt_wind != nt_wave:
    print(f"Warning: wind and wave time series have different lengths (wind={nt_wind}, wave={nt_wave}). Truncating to {nt} samples.")

# Truncate arrays to common length if necessary
wind_t = windForce["t"][:nt]
wind_F = windForce["F"][:nt]
wind_M = windForce["M"][:nt]
wave_t = waveForce["t"][:nt]
wave_F = waveForce["F"][:nt]
wave_M = waveForce["M"][:nt]

totalForce = {
    "t": wind_t,
    "F": wind_F + wave_F,
    "M": wind_M + wave_M,
}


# Save the results for later use
# Ensure the directory used by get_output_file() exists (was previously creating a local `savedStates` folder)
output_folder = os.path.dirname(get_output_file("wave_data.json"))
os.makedirs(output_folder, exist_ok=True)
saveToJSON(wave_data, get_output_file("wave_data.json"))
saveToJSON(wind_data, get_output_file("wind_data.json"))


# ============================================================================
# Plots: Wind and wave forces, and total overturning moment
# ============================================================================
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


# ============================================================================
# Statistics of the wind speed time series and the total wind force time series
# # ============================================================================

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

print(f"\nTotal Combined Force:")
print(f"  Mean: {np.mean(totalForce['F'])/1e6:.3f} MN")
print(f"  Std: {np.std(totalForce['F'])/1e6:.3f} MN")
print(f"  Min: {np.min(totalForce['F'])/1e6:.3f} MN")
print(f"  Max: {np.max(totalForce['F'])/1e6:.3f} MN")