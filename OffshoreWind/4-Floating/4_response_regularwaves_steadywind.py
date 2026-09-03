" Response to Regular Waves and Steady Wind "

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
# Load the variables, timeInfo, and SparBuoyData
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
waves = loadFromJSON(get_input_file("wave_regular.json"))
waves["z"] = z

with Timer("wind"):
    # Wind speed - should be constant
    wind = loadFromJSON(get_input_file("wind_steady.json")) # Make sure the is not turbulence (I) in the wind speed file
    wind.update(timeInfo)
    wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 1)
    wind = calculateWindTimeSeriesFFT(wind)

# Wave kinematics - should be zero
with Timer("waves"):
    # calculate the wave kinematics - zero at this stage
    waves.update(timeInfo)
    waves["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    waves = calculateRegularWaveFrequencyInformation(waves)
    waves = calculateFreeSurfaceElevationTimeSeries(waves)
    waves = calculateKinematics(waves)


# ============================================================================
# Get the Response to Regular Waves and Steady Wind
# ============================================================================

# Set the drag coefficient for the spar buoy
SparBuoyData["CD"] = 0.

# Get the response to regular waves and steady wind for pitch decay
with Timer("Integration"):
    q0 = np.array([0,0,0,0,np.nan])              # Initial conditions for surge decay
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

# Initialize the response dictionary
response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]


# Plot the response and get the figures, results
fig_response = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
recolor_lines_by_time(fig_response, t_split=600.0)      # Recolor lines with time split at 600s

fig_response[0,0].figure.savefig(get_output_figure("Response_RegularWaves_SteadyWind.pdf"))
print(f"Response to Regular Waves and Steady Wind")
print(f'Surge Standard deviation [m]: {np.std(q[:,0]):.3f}')
print(f'Surge Mean [m]: {np.mean(q[:,0]):.3f}')
print(f'Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1])):.3f}')
print(f'Pitch Mean [deg]: {np.rad2deg(np.mean(q[:,1])):.3f}')