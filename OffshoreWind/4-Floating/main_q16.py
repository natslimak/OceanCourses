import numpy as np
import pylab as plt
import os
import sys  

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..','functions', 'python'))
sys.path.append(helpers_path)

from common import *
from waves import *
from wind import *
from integration import ode4
from floaterIntegration import dqdt
from plotting import makeplots 

inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)
ov = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', 'outputVariables', x)


of = "outputFig"
os.makedirs(of, exist_ok=True)
def ofy(fileName):
    return os.path.join(of, fileName)

# Close any existing plots
plt.close('all')

# %% ----Q16: RESPONSE TO WIND--------------------------------------

# Load the time information
timeInfo = loadFromJSON(fp('time.json'))

# Wind speed - should be constant
# Just a loop to handle both wind cases
winds = ['wind16B', 'wind16A']
colors = ['g', 'b']
labels = ['16 m/s', '10 m/s']

# Initialize the figure outside the loop
#fig16 = plt.figure()
#plt.hold(True)  # Keep the plot open for multiple plots
# Create figure and axes once
fig16, ax16 = plt.subplots(4,2, sharex='col')

for i in range(len(winds)):
    wind = loadFromJSON(fp(f'{winds[i]}.json'))
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind["V_hub"] = np.zeros_like(wind["t"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 1)  
    wind = calculateWindTimeSeriesFFT(wind)

    # Load the rotor and other necessary parameters
    IEA22MWRotor = loadFromJSON(fp('iea22mw.json'))
    IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
    IEA22MWRotor['gamma'] = 0.0
    IEA22MWRotor['active'] = True

    # Disable drag forcing
    SparBuoyData = loadFromJSON(ov("SparBuoyDataComplete.json"))
    SparBuoyData['CD'] = 0  # Dry decay test

    # Vertical locations along floater
    z = np.linspace(SparBuoyData['z_Bot'], 0, 100)
    SparBuoyData['z'] = z

    # Calculate the wave kinematics from Irregular wave
    waves = loadFromJSON(fp('nowaves.json'))
    waves['z'] = z
    random_seed_waves = 1
    waves.update(timeInfo)
    waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
    # fixme: Add wave calculations here (use 4 functions if applicable)
    waves['eta'] = np.zeros_like(waves['t'])        # free surface elevation
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
    fig16 = makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[i], ax=ax16)

    
    # Print the standard deviations
    print(f'Q16 Surge Standard deviation [m]: {np.std(q[:,0])}')
    print(f'Q16 Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1]))}')

fig16[0,0].legend(labels, fontsize=7, loc='upper right')             
fig16[0,0].figure.savefig(ofy("fig16.pdf"))



# Add labels and legend
#plt.legend(labels)  # Use labels from the loop
