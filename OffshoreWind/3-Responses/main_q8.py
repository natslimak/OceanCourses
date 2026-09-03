''' Slow and Fast Wave and Wind Calculations for Response Analysis'''


import os
import sys 

# Add the function folder to the path so the helper modules can be imported.
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeries, calculateKinematics, calculateFreeSurfaceElevationTimeSeriesFFT, calculateKinematicsFFT
from wind import calculateWindTimeSeries, calculateWindTimeSeriesFFT
from common import loadFromJSON, Timer
import pylab as plt
import os.path
import numpy as np


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', '..', output_dir, fname)



# WAVES
waves = loadFromJSON(get_input_file("waves4.json"))
timeInfo = loadFromJSON(get_input_file("time.json"))
waves.update(timeInfo)
waves["t"] = np.arange(0., waves["TDur"], waves["dt"])
waves = calculateJONSWAPSpectrum(waves)

# Calculate the wave kinematics from Irregular wave with FFT and without FFT
with Timer("slow waves"):
    waves = calculateFreeSurfaceElevationTimeSeries(waves)
    waves = calculateKinematics(waves)

with Timer("fast waves"):
    wavesFast = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    wavesFast = calculateKinematicsFFT(wavesFast)


   
# WIND
wind = loadFromJSON(get_input_file("wind4.json"))
wind.update(timeInfo)

# Calculate the wind time series with FFT and without FFT
with Timer("Slow wind"):
    wind = calculateWindTimeSeries(wind)
    
with Timer("Fast wind"):
    windFast = calculateWindTimeSeriesFFT(wind)    



# ============================================================================
# Plot the results
# ============================================================================

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