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

# %% ----Q13: RESPONSE TO IRREGULAR WAVES--------------------------------------

# Load the variables
timeInfo = loadFromJSON(fp("time.json"))
constants = loadConstants()
SparBuoyData = loadFromJSON(ov("SparBuoyDataComplete.json"))

# Load the rotor data
IEA22MWRotor = loadFromJSON(fp("iea22mw.json"))
# Controller parameter & state of the rotor
IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
IEA22MWRotor['gamma'] = 0.0
IEA22MWRotor['active'] = True

# Wind speed - should be constant
# We set up a spectrum with zero turbulence.
# Just an easy way to reuse existing functions
random_seed_wind = 2
wind = loadFromJSON(fp("wind15.json"))
wind.update(timeInfo)
wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
wind["V_hub"] = np.zeros_like(wind["t"])

# fixme: Add WIND calculations here (use 3 functions if applicable)
wind = calculateKaimalSpectrum(wind)
wind = generateRandomPhases(wind, 2)
wind = calculateWindTimeSeriesFFT(wind)


# Vertical locations along floater
z = np.linspace(SparBuoyData['z_Bot'], 0, 100)
SparBuoyData['z'] = z

# Calculate the wave kinematics from Irregular wave
waves = loadFromJSON(fp("wave13.json"))
waves['z'] = z
random_seed_waves = 1
waves.update(timeInfo)
waves['t'] = np.arange(0, waves['TDur'], waves['dt'])

# fixme: Add wave calculations here (use 4 functions if applicable)
'''
waves['eta'] = np.zeros_like(waves['t'])        # free surface elevation
waves['u']   = np.zeros((len(waves['t']), len(waves['z'])))  # horizontal velocity at each z
waves['ut']  = np.zeros((len(waves['t']), len(waves['z'])))  # vertical velocity at each z
'''
waves = calculateJONSWAPSpectrum(waves)
waves = generateRandomPhases(waves, 1)
waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
waves = calculateKinematicsFFT(waves)

# Response
tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
q0 = np.array([0, 0, 0, 0, np.nan])
q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode
response["x1"] = q[:, 0]
response["x5"] = q[:, 1]

# Plotting results
fig15 = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
fig15[0,0].figure.savefig(ofy("fig15.pdf"))


# --- make the line colors change at t=600s
t_start = response["t"][0]
t_split = 600.0
t_end = response["t"][-1]
axs = np.array(fig15)
left_col = axs[:, 0].ravel()
# Recolor plotted lines so the line color changes at t_split
for ax in left_col:
    # Copy list because we'll remove original lines while iterating
    orig_lines = list(ax.get_lines())
    for line in orig_lines:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        if x.size == 0:
            continue

        # masks for the two time segments
        mask1 = x <= t_split
        mask2 = x > t_split

        # original line properties to preserve style
        lw = line.get_linewidth()
        ls = line.get_linestyle()
        mk = line.get_marker()
        msz = line.get_markersize()
        alpha = line.get_alpha()
        orig_label = line.get_label()

        # Plot first segment (red) if present
        if mask1.any():
            label1 = orig_label if (mask1.any() and not mask2.any()) else None
            ax.plot(x[mask1], y[mask1], color='red', linewidth=lw, linestyle=ls, marker=mk, markersize=msz, alpha=alpha, label=label1)

        # Plot second segment (blue) if present
        if mask2.any():
            label2 = orig_label if (mask2.any() and not mask1.any()) else None
            ax.plot(x[mask2], y[mask2], color='blue', linewidth=lw, linestyle=ls, marker=mk, markersize=msz, alpha=alpha, label=label2)

        # remove the original (single-color) line
        try:
            line.remove()
        except Exception:
            pass
            
left_col[0].figure.savefig(ofy("fig15.pdf"))        
fig15[0,0].figure.savefig(ofy("fig15.pdf"))

# Calculate and print standard deviations
surge_std = np.std(q[:, 0])
pitch_std = np.rad2deg(np.std(q[:, 1]))

print(f'Q15 Surge Standard deviation [m]: {surge_std}')
print(f'Q15 Pitch Standard deviation [deg]: {pitch_std}')
