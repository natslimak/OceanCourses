""" Overturning moment at seabed (mudline) for irregular waves. """

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the functions folder to the path 
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON, generateRandomPhases
from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeries, calculateKinematics
from monopile import forceIntegrate


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)

# Load monopile, sea-state and time properties (period, amplitude, etc.)
monopile_props = loadFromJSON(get_input_file("monopile.json"))
wave_data = loadFromJSON(get_input_file("wave_irregular.json"))
time_data = loadFromJSON(get_input_file("time.json"))
wave_data.update(time_data)
wave_data["t"] = np.arange(0., wave_data["TDur"], wave_data["dt"])

# Get wave parameters (wavelength, wavenumber, etc.) and compute the JONSWAP spectrum
wave_data = calculateJONSWAPSpectrum(wave_data)
randomSeedWaves = 1
wave_data = generateRandomPhases(wave_data, seed=randomSeedWaves)

# Recompute kinematics to ensure fields exist locally
wave_data = calculateFreeSurfaceElevationTimeSeries(wave_data)
wave_data = calculateKinematics(wave_data)


# ============================================================================
# Caculate forces at seabed (mudline)
# ============================================================================
force = dict()
force["t"] = wave_data["t"]
force["F"], force["M"] = np.zeros_like(wave_data["t"]), np.zeros_like(wave_data["t"])

for i_, t_ in enumerate(wave_data["t"]):
    force["F"][i_], force["M"][i_]  = forceIntegrate(monopile_props, wave_data["u"][i_,:], wave_data["ut"][i_,:],
        wave_data["z"], 0., wave_data["z"][0]) # Moment at seabed


# ============================================================================
# Calulating overturning moment for Hs = stddev*4
# ============================================================================

# After calculating the wave spectrum and time series, add validation:
# Check if 4*sigma_eta = Hs
sigma_eta = np.std(wave_data["eta"])
expected_sigma = wave_data["Hs"] / 4.0
scaling_factor = expected_sigma / sigma_eta

print(f"Current σ_η: {sigma_eta:.3f}")
print(f"Expected σ_η (Hs/4): {expected_sigma:.3f}")
print(f"Scaling factor needed: {scaling_factor:.3f}")

# Apply scaling if needed for the reasonable match of the standard deviation to Hs/4
if abs(scaling_factor - 1.0) > 0.01:  # If scaling is significantly needed   -> could have been also directly scaled the AMPLITUDE spectrum
     wave_data["eta"] *= scaling_factor
     # Also scale velocities and accelerations accordingly
     wave_data["u"] *= scaling_factor
     wave_data["ut"] *= scaling_factor
     print(f"Applied scaling factor: {scaling_factor:.3f}")

# Verify the scaling
sigma_eta_corrected = np.std(wave_data['eta'])
print(f"Corrected σ_η: {sigma_eta_corrected:.3f}")
print(f"Ratio 4σ_η/Hs: {4*sigma_eta_corrected/wave_data['Hs']:.3f}")



# ============================================================================
# Plots: Overturning moment and wave elevation
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top subplot - Overturning Moment
ax1.plot(force["t"], force["M"])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Overturning Moment [Nm]')
ax1.set_title('Overturning Moment at Mudline')
ax1.grid(True)

# Bottom subplot - Wave Elevation
ax2.plot(wave_data["t"], wave_data["eta"])
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Wave Elevation [m]')
ax2.set_title('Wave Elevation at Sea Surface')
ax2.grid(True)

plt.tight_layout()
plt.show()


# ============================================================================
# Statistical Analysis
# ============================================================================
print("Wave elevation statistics:")
print(f"  Mean: {np.mean(wave_data['eta']):.4f} m")
print(f"  Std: {np.std(wave_data['eta']):.4f} m")
print(f"  Min: {np.min(wave_data['eta']):.4f} m")
print(f"  Max: {np.max(wave_data['eta']):.4f} m")


print("Overturning moment statistics:")
print(f"  Mean: {np.mean(force['M'])/1e6:.4f} MNm")
print(f"  Std: {np.std(force['M'])/1e6:.4f} MNm")
print(f"  Min: {np.min(force['M'])/1e6:.4f} MNm")
print(f"  Max: {np.max(force['M'])/1e6:.4f} MNm")

