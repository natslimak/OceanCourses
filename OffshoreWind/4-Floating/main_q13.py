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
plt.close('all')

#%% Q13: response to waves and steady wind

# Load the variables here
timeInfo = loadFromJSON(fp("time.json"))
constants = loadConstants()
SparBuoyData = loadFromJSON(ov("SparBuoyDataComplete.json"))
SparBuoyData["CD"] = 0.6; 

z = np.linspace(SparBuoyData["z_Bot"], 0., 100)
SparBuoyData["z"] = z
# Dry decay test
SparBuoyData["CD"] = 0.

# Wave kinematics - should be zero
waves = loadFromJSON(fp("wave12.json"))
waves["z"] = z

with Timer("wind"):
    # Wind speed - should be constant
    # FIXME Modify the input file wind13.json so that it represents a steady wind
    wind = loadFromJSON(fp("wind13.json"))
    wind.update(timeInfo)
    wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 1)
    wind = calculateWindTimeSeriesFFT(wind)

# Load the rotor
IEA22MWRotor = loadFromJSON(fp("iea22mw.json"))
IEA22MWRotor["ARotor"] = 0.25*np.pi*IEA22MWRotor["DRotor"]**2

# Controller parameter & state of the rotor
IEA22MWRotor["gamma"] = 0.
IEA22MWRotor["active"] = True

with Timer("waves"):
    # calculate the wave kinematics - zero at this stage
    waves.update(timeInfo)
    waves["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    waves = calculateRegularWaveFrequencyInformation(waves)
    waves = calculateFreeSurfaceElevationTimeSeries(waves)
    waves = calculateKinematics(waves)

# Integration time array
tode = np.arange(0., timeInfo["TDur"], 2*timeInfo["dt"])

# Response
# Pitch decay with wind
with Timer("Integration"):
    # Initial conditions for surge decay
    q0 = np.array([0,0,0,0,np.nan])
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]

fig13 = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')

# --- make the line colors change at t=600s
t_start = response["t"][0]
t_split = 600.0
t_end = response["t"][-1]
axs = np.array(fig13)
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

left_col[0].figure.savefig(ofy("fig13.pdf"))        
fig13[0,0].figure.savefig(ofy("fig13.pdf"))
#%% Import the results from Q12
'''from main_q12 import response as response_waveonly
fig13 = makeplots(wind, waves, SparBuoyData, response_waveonly, timeInfo, 'b', ax=fig13)

'''

print(f'Q13 Surge Standard deviation [m]: {np.std(q[:,0])}')
print(f'Q13 Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1]))}')