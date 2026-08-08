''' Dynamic loads on a monopile under wind and wave loading.'''

import os
import sys 
import numpy as np
import pylab as plt

# Add the function folder to the path so the helper modules can be imported.
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)


from common import loadConstants, loadFromJSON, downsample, saveToJSON    
from loads import calculateStaticWindLoads, calculateStaticWaveLoads, calculateDynamicLoads
from monopile import computeElementwiseQuantities


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', '..', output_dir, fname)

# Load the structural motions
q = loadFromJSON(get_output_file("q.json"))
alphaDotDot = np.gradient(q["alphaDot"], q["t"])
q["alphaDotDot"] = alphaDotDot

# Load the wind, waves, monopile and the rotor
wind = loadFromJSON(get_output_file("wind_data.json"))
waves = loadFromJSON(get_output_file("wave_data.json"))
iea22mw = loadFromJSON(get_input_file("iea22mw.json"))
monopile_props = loadFromJSON(get_input_file("monopile.json"))


# ============================================================================
# Calculate the static loads
# ============================================================================

# Calculate the static wind loads
windDownsampled = downsample(wind, dropEvery=2, listOfFields=["t", "V_hub"])
windLoads = calculateStaticWindLoads(windDownsampled, iea22mw, monopile_props, q)

# Calculate the static wave loads
wavesDownsampled = downsample(waves, dropEvery=2, listOfFields=["t", "u", "ut"])
waveLoads = calculateStaticWaveLoads(wavesDownsampled, monopile_props, q)

# Calculate the dynamic loads
monopile_props = computeElementwiseQuantities(monopile_props)
dynamicLoads = calculateDynamicLoads(monopile_props, q)

saveToJSON(dynamicLoads, "savedStates/dynLoads.json")


# ============================================================================
# Plots: Dynamic Loads on the Monopile
# ============================================================================
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


# ============================================================================
# Statistical Analysis of Loads
# ============================================================================
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