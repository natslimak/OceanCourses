""" Dynamic response of a monopile under wave and wind loading over time."""

import os
import sys 
import pylab as plt
import numpy as np

# Add the function folder to the path so the helper modules can be imported.
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from integration import ode4, dqdt
from common import loadFromJSON, saveToJSON


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', '..', output_dir, fname)

# Time vector for the simulation: from 0 s to 660 s with a 0.1 s time step.
tIntegration = np.arange(0., 600., 0.1)

# Load the monopile, wind and wave data, as well as the time properties
monopile_props = loadFromJSON(get_input_file("monopile.json"))
wave_data = loadFromJSON(get_output_file("wave_data.json"))
wind_data = loadFromJSON(get_output_file("wind_data.json"))
iea22mw = loadFromJSON(get_input_file("iea22mw.json"))
iea22mw["ARotor"] = iea22mw["DRotor"]**2*np.pi*0.25


# ============================================================================
# Compute the response of the monopile under wave and wind loading
# ============================================================================

# Calculate the response variables for the monopile over time.
q0 = np.array([1.0,0.])
q = ode4(dqdt, tIntegration, q0, monopile_props, iea22mw,
                wave_data, wind_data, rotor_state="on")

# Convert the generalized response into top displacement and top velocity.
# phiNodalTop is the modal shape evaluated at the top of the monopile.
phiNodalTop = monopile_props["phiNodal"][-1]
xTTop = q[:,0]*phiNodalTop
xDotTTop = q[:,1]*phiNodalTop

# Save the response variables to a JSON file for later use or post-processing.
alpha = dict()
alpha["t"] = tIntegration
alpha["alpha"] = q[:,0]
alpha["alphaDot"] = q[:,1]
saveToJSON(alpha, get_output_file("q.json"))


# ============================================================================
# Plots: Monopile Response Over Time
# ============================================================================
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


# ============================================================================
# Statistics computed after the transient period (t >= 60 s)
# ============================================================================
mask = tIntegration >= 60
xTTop_60 = xTTop[mask]
xDotTTop_60 = xDotTTop[mask]

# Print summary statistics for the displacement and velocity after 60 s.
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
