" Response to Steady Wind above and below Rated Power of the Rotor "

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
from plotting import makeplots 


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
# SETUP: Load other important Files and initialize the plot
# ============================================================================
# Load the time information
timeInfo = loadFromJSON(get_input_file('time.json'))

# Wind speed - should be constant
winds = ['_wind_above_rated', '_wind_below_rated']
colors = ['g', 'b']
labels = ['16 m/s', '10 m/s']

# Create figure and axes once
fig_response, ax16 = plt.subplots(4,2, sharex='col')


# ============================================================================
# Get the Responses to two different Wind Speeds 
# ============================================================================
for i in range(len(winds)):

    # Load the wind conditions
    wind = loadFromJSON(get_input_file(f'{winds[i]}.json'))
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind["V_hub"] = np.zeros_like(wind["t"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 1)  
    wind = calculateWindTimeSeriesFFT(wind)

    # Load the rotor and other necessary parameters
    IEA22MWRotor = loadFromJSON(get_input_file('iea22mw.json'))
    IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
    IEA22MWRotor['gamma'] = 0.0
    IEA22MWRotor['active'] = True

    # Disable drag forcing (no waves)
    SparBuoyData = loadFromJSON(get_output_file("SparBuoyDataComplete.json"))
    SparBuoyData['CD'] = 0  # Dry decay test

    # Vertical locations along floater
    z = np.linspace(SparBuoyData['z_Bot'], 0, 100)
    SparBuoyData['z'] = z

    # Calculate the wave kinematics from Irregular wave
    waves = loadFromJSON(get_input_file('nowaves.json'))
    waves['z'] = z
    random_seed_waves = 1
    waves.update(timeInfo)
    waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
    waves['eta'] = np.zeros_like(waves['t'])                     # free surface elevation
    waves['u']   = np.zeros((len(waves['t']), len(waves['z'])))  # horizontal velocity at each z
    waves['ut']  = np.zeros((len(waves['t']), len(waves['z'])))  # vertical velocity at each z

    # Response
    tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
    q0 = np.array([0, 0, 0, 0, np.nan])
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]

    # Plot the results on the same figure
    fig_response = makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[i], ax=ax16)

    # Print the standard deviations and means
    print(f"\nResponse to Steady Wind {labels[i]} m/s")
    print(f'Surge Standard deviation [m]: {np.std(q[:,0]):.3f}')
    print(f'Surge Mean [m]: {np.mean(q[:,0]):.3f}')
    print(f'Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1])):.3f}')
    print(f'Pitch Mean [deg]: {np.rad2deg(np.mean(q[:,1])):.3f}')


# Plot the legend and save the figure
fig_response[0,0].legend(labels, fontsize=7, loc='upper right')             
fig_response[0,0].figure.savefig(get_output_figure("Response_Comparison_Above&Below_Rated.pdf"))


