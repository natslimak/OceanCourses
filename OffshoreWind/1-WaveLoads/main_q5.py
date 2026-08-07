import os 
import sys
import pandas as pd

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)

# Load the results from main_q2.py
#from main_q2 import wavesQ2, forceQ2, monopileDict
import numpy as np
import matplotlib.pyplot as plt
from waves import *
from common import loadFromJSON
from monopile import forceIntegrate


# Load the monopile information
monopileDict = loadFromJSON(fp("monopile.json"))

# Load the information for the creation of the wave
wavesQ5 = loadFromJSON(fp("wave5.json"))

# Load the time discretization info
timeQ5 = loadFromJSON(fp("time.json"))
wavesQ5.update(timeQ5)
wavesQ5["t"] = np.arange(0., wavesQ5["TDur"], wavesQ5["dt"])

wavesQ5 = calculateJONSWAPSpectrum(wavesQ5)

plt.plot(wavesQ5["f"], wavesQ5["Spectrum"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Spectral Density")
plt.title("JONSWAP Spectrum")
plt.show()

wavesQ5 = calculateKinematics(wavesQ5)
wavesQ5 = calculateFreeSurfaceElevationTimeSeries(wavesQ5)


plt.plot(wavesQ5["t"], wavesQ5["eta"])
plt.xlabel("Time [s]")
plt.ylabel("Free Surface Elevation [m]")
plt.title("Free surface elevation")
plt.show()


forceQ5Wheeler = dict()
forceQ5Wheeler["t"] = wavesQ5["t"]
forceQ5Wheeler["F"], forceQ5Wheeler["M"] = np.zeros_like(wavesQ5["t"]), np.zeros_like(wavesQ5["t"])

# Calculate zPhys
wavesQ5["z_phys"] = wavesQ5["z"][None,:] + wavesQ5["eta"][:, None]*(1+wavesQ5["z"][None,:]/wavesQ5["h"])


for i_, t_ in enumerate(wavesQ5["t"]):
    u_this_step = wavesQ5["u"][i_,:]
    ut_this_step = wavesQ5["ut"][i_,:]
    z_phys_this_step = wavesQ5["z_phys"][i_,:]
    
    F, M  = forceIntegrate(monopileDict, u_this_step , ut_this_step,z_phys_this_step, 0.)
    forceQ5Wheeler["F"][i_] = F 
    forceQ5Wheeler["M"][i_] = M

fig, axs = plt.subplots(2, 1)

# Force time series
axs[0].plot(forceQ5Wheeler["t"], forceQ5Wheeler["F"])
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Force [N]")
axs[0].set_title("Force on monopile with JONSWAP spectrum")

# Moment time series QUESTION 6
axs[1].plot(forceQ5Wheeler["t"], forceQ5Wheeler["M"])
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Moment around mudline [Nm]")
axs[1].set_title("Moment on monopile with JONSWAP spectrum")

plt.tight_layout()
plt.show()

# Compute descriptive statistics
stats = {
    "Mean":     [np.mean(wavesQ5["eta"]),      np.mean(forceQ5Wheeler["F"]),    np.mean(forceQ5Wheeler["M"])],
    "Std":      [np.std(wavesQ5["eta"], ddof=0),np.std(forceQ5Wheeler["F"], ddof=0), np.std(forceQ5Wheeler["M"], ddof=0)],
    "Max":      [np.max(wavesQ5["eta"]),       np.max(forceQ5Wheeler["F"]),     np.max(forceQ5Wheeler["M"])],
    "Min":      [np.min(wavesQ5["eta"]),       np.min(forceQ5Wheeler["F"]),      np.min(forceQ5Wheeler["M"])]
}
table = pd.DataFrame(stats, index=["Free Surface Elevation [m]", "Wheeler Force [N]", "Wheeler Moment [Nm]"])

# Display table in console
print("\nTable with mean, standard deviation, min and max values\n")
print(table)

sigma_eta = table.loc["Free Surface Elevation [m]", "Std"]
Hs_from_calc = 4.0 * sigma_eta
Hs_from_given = wavesQ5["Hs"]
print(f"Given significant wave height: {Hs_from_given}m and the calculated one: {Hs_from_calc}m")

# histogram for eta
plt.figure()
plt.hist(wavesQ5["eta"], bins=80)
plt.xlabel("η [m]")
plt.ylabel("Counts")
plt.title("Histogram of free-surface elevation η")
plt.show()