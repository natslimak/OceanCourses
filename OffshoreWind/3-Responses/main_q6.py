import os
import sys 

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','functions', 'python'))
sys.path.append(helpers_path)

import numpy as np
from integration import ode4, dqdt
from common import loadFromJSON, saveToJSON
import os.path
import pylab as plt

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

tIntegration = np.arange(0., 600., 0.1)

# Load the dictionaries
monopileDict = loadFromJSON(fp("monopile.json"))
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi*0.25
waves5 = loadFromJSON(ss("waves4.json"))
wind5 = loadFromJSON(ss("wind4.json"))


# Perform the integration
q0 = np.array([1.0,0.])
# FIXME Assignment 3 Q1.6: fix the dqdt function in integration.py
q = ode4(dqdt, tIntegration, q0, monopileDict, iea22mw,
                waves5, wind5, rotor_state="on")

phiNodalTop = monopileDict["phiNodal"][-1]
xTTop = q[:,0]*phiNodalTop
xDotTTop = q[:,1]*phiNodalTop

# Plot the results
f,ax = plt.subplots(2, sharex=True, figsize=(10, 6))
ax[0].plot(tIntegration, xTTop)
ax[1].plot(tIntegration, xDotTTop)
ax[0].grid();ax[1].grid()
ax[1].set_xlabel("t[s]")
ax[0].set_ylabel(r"$X_{Top}$ [m]")
ax[1].set_ylabel(r"$\dot X_{Top}$ [m/s]")
ax[0].set_title("Top displacement of monopile")
ax[1].set_title("Top velocity of monopile")
for a in np.array(ax).flatten():
    a.axvline(60.0, color='k', linestyle='--', linewidth=1, zorder=2)
ymax = ax[0].get_ylim()[1]
ax[0].text(60.0, ymax*0.90, '60 s', color='k', ha='left', va='top',
             fontsize=9, backgroundcolor=None)
plt.show()

alpha = dict()
alpha["t"] = tIntegration
alpha["alpha"] = q[:,0]
alpha["alphaDot"] = q[:,1]

saveToJSON(alpha, ss("q.json"))

# ----------------------------------------------
# Statistics
# ----------------------------------------------
mask = tIntegration >= 60
xTTop_60 = xTTop[mask]
xDotTTop_60 = xDotTTop[mask]

# Now compute stats only for data after 60s
print("\nXT Top (t >= 60 s):")
print(f"  Mean: {np.mean(xTTop_60):.3f} m")
print(f"  Std: {np.std(xTTop_60):.3f} m")
print(f"  Min: {np.min(xTTop_60):.3f} m")
print(f"  Max: {np.max(xTTop_60):.3f} m")

print(f"\nXDotT Top (t >= 60 s):")
print(f"  Mean: {np.mean(xDotTTop_60):.3f} m/s")
print(f"  Std: {np.std(xDotTTop_60):.3f} m/s")
print(f"  Min: {np.min(xDotTTop_60):.3f} m/s")
print(f"  Max: {np.max(xDotTTop_60):.3f} m/s")
