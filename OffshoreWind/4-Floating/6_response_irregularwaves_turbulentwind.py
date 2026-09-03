" Response to Irregular Waves and Steady Wind (FFT) "

import numpy as np
import pylab as plt
import os
import sys  

# Add the function folder to the path so the helper modules can be imported.
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import *
from waves import *
from wind import *
from integration import ode4
from floaterIntegration import dqdt
from plotting import makeplots, recolor_lines_by_time


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
figure_dir = "outputFig"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', output_dir, fname)
get_output_figure = lambda fname: os.path.join(os.path.dirname(__file__), figure_dir, fname)


# ============================================================================
# SETUP: Load other important Files 
# ============================================================================
# Load the variables
constants = loadConstants()
timeInfo = loadFromJSON(get_input_file("time.json"))
SparBuoyData = loadFromJSON(get_output_file("SparBuoyDataComplete.json"))

# Set up the z array for the spar buoy
z = np.linspace(SparBuoyData["z_Bot"], 0., 100)
SparBuoyData["z"] = z

# Load the rotor
IEA22MWRotor = loadFromJSON(get_input_file("iea22mw.json"))
IEA22MWRotor["ARotor"] = 0.25*np.pi*IEA22MWRotor["DRotor"]**2
IEA22MWRotor["gamma"] = 0.          # Controller parameter
IEA22MWRotor["active"] = True       # State of the rotor

# Integration time array
tode = np.arange(0., timeInfo["TDur"], 2*timeInfo["dt"])


# ============================================================================
# SETUP: Load the wave and wind conditions
# ============================================================================
# Wind speed - should be constant with turbulence
wind = loadFromJSON(get_input_file("wind_turbulent.json"))
wind.update(timeInfo)
wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
wind["V_hub"] = np.zeros_like(wind["t"])
wind = calculateKaimalSpectrum(wind)
wind = generateRandomPhases(wind, 2)
wind = calculateWindTimeSeriesFFT(wind)


# Calculate the wave kinematics from Irregular wave
waves = loadFromJSON(get_input_file("wave_irregular.json"))
waves['z'] = z
waves.update(timeInfo)
waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
waves = calculateJONSWAPSpectrum(waves)
waves = generateRandomPhases(waves, 1)
waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
waves = calculateKinematicsFFT(waves)

# ============================================================================
# Get the Response to Irregular Waves and Steady Wind
# ============================================================================

# Response
q0 = np.array([0, 0, 0, 0, np.nan])
q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

# Initialize the response dictionary 
response = dict()
response["t"] = tode
response["x1"] = q[:, 0]
response["x5"] = q[:, 1]

# Plot the response and get the figures, results
fig_response = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
recolor_lines_by_time(fig_response, t_split=600.0)      # Recolor lines with time split at 600s

fig_response[0,0].figure.savefig(get_output_figure("Response_IrregularWaves_TurbulentWind.pdf"))
print(f"Response to Irregular Waves and Turbulent Wind")
print(f'Surge Standard deviation [m]: {np.std(q[:,0]):.3f}')
print(f'Surge Mean [m]: {np.mean(q[:,0]):.3f}')
print(f'Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1])):.3f}')
print(f'Pitch Mean [deg]: {np.rad2deg(np.mean(q[:,1])):.3f}')