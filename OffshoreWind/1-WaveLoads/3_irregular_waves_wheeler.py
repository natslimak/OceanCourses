"""" Monopile Loads for a JONSWAP Sea State with Wheeler-stretched Depths"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add functions folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON, generateRandomPhases
from waves import calculateJONSWAPSpectrum, calculateKinematics, calculateFreeSurfaceElevationTimeSeries
from monopile import forceIntegrate


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)

# Load monopile, sea-state and time properties (period, amplitude, etc.)
monopile_props = loadFromJSON(get_input_file("monopile.json"))
wave_data = loadFromJSON(get_input_file("wave_irregular.json"))
time_data = loadFromJSON(get_input_file("time.json"))
wave_data.update(time_data)
wave_data["t"] = np.arange(0., wave_data["TDur"], wave_data["dt"])

# Get wave parameters (wavelength, wavenumber, etc.) and compute the JONSWAP spectrum
wave_data = calculateJONSWAPSpectrum(wave_data)
rp_dict = generateRandomPhases(wave_data, seed=42)
wave_data["randomPhases"] = rp_dict["randomPhases"]

# Recompute kinematics to ensure fields exist locally
wave_data = calculateKinematics(wave_data)
wave_data = calculateFreeSurfaceElevationTimeSeries(wave_data)


# ============================================================================
# Spectrum, kinematics and free surface
# ============================================================================

# Plot the JONSWAP spectrum
plt.figure()
plt.plot(wave_data["f"], wave_data["Spectrum"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Spectral Density")
plt.title("JONSWAP Spectrum")
plt.grid(alpha=0.3)
plt.show()

# Plot the free-surface elevation time series
plt.figure()
plt.plot(wave_data["t"], wave_data["eta"])
plt.xlabel("Time [s]")
plt.ylabel("Free Surface Elevation [m]")
plt.title("Free-surface elevation")
plt.grid(alpha=0.3)
plt.show()


# ============================================================================
# Wheeler stretching: compute physical depths z_phys(t,z)
# and interpolate velocity/acceleration onto those depths per timestep
# ============================================================================
z_ref = wave_data["z"]            # reference depths (nz,)
eta = wave_data["eta"]            # free surface time series (nt,)
h = wave_data["h"]

# z_phys shape (nt, nz)
z_phys = z_ref[None, :] + eta[:, None] * (1.0 + z_ref[None, :] / h)
wave_data["z_phys"] = z_phys

nt = wave_data["t"].shape[0]
force_nominal = {"t": wave_data["t"], "F": np.zeros(nt), "M": np.zeros(nt)}
force_wheeler = {"t": wave_data["t"], "F": np.zeros(nt), "M": np.zeros(nt)}

# Interpolate velocities onto z_phys each timestep and compute forces
for ti in range(nt):
    u_ref = wave_data["u"][ti, :]
    ut_ref = wave_data["ut"][ti, :]

    # Nominal (use reference depths)
    Fn, Mn = forceIntegrate(monopile_props, u_ref, ut_ref, z_ref, 0.0)
    force_nominal["F"][ti] = Fn
    force_nominal["M"][ti] = Mn

    # Interpolate onto Wheeler physical depths (z_phys may not be monotonic; np.interp requires increasing xp)
    # Ensure xp (z_ref) is strictly increasing for np.interp (it should be). If not, sort.
    if np.all(np.diff(z_ref) > 0):
        u_phys = np.interp(z_phys[ti, :], z_ref, u_ref)
        ut_phys = np.interp(z_phys[ti, :], z_ref, ut_ref)
    else:
        # fallback: sort reference depths and corresponding values
        idx_sort = np.argsort(z_ref)
        z_sorted = z_ref[idx_sort]
        u_sorted = u_ref[idx_sort]
        ut_sorted = ut_ref[idx_sort]
        u_phys = np.interp(z_phys[ti, :], z_sorted, u_sorted)
        ut_phys = np.interp(z_phys[ti, :], z_sorted, ut_sorted)

    Fw, Mw = forceIntegrate(monopile_props, u_phys, ut_phys, z_phys[ti, :], 0.0)
    force_wheeler["F"][ti] = Fw
    force_wheeler["M"][ti] = Mw


# ============================================================================
# Plots: compare Wheeler vs nominal results and show moment
# ============================================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(force_nominal["t"], force_nominal["F"], label="Nominal", linewidth=1)
axs[0].plot(force_wheeler["t"], force_wheeler["F"], label="Wheeler", linewidth=1, linestyle='--')
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Horizontal Force [N]")
axs[0].set_title("Horizontal Force on monopile (JONSWAP sea state)")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(force_wheeler["t"], force_wheeler["M"], color='C1')
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Moment about mudline [Nm]")
axs[1].set_title("Moment on monopile (Wheeler-stretched)")
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# Descriptive statistics (using Wheeler results as before)
# ============================================================================
stats = {
    "Mean": [np.mean(wave_data["eta"]), np.mean(force_wheeler["F"]), np.mean(force_wheeler["M"])],
    "Std":  [np.std(wave_data["eta"], ddof=0), np.std(force_wheeler["F"], ddof=0), np.std(force_wheeler["M"], ddof=0)],
    "Max":  [np.max(wave_data["eta"]), np.max(force_wheeler["F"]), np.max(force_wheeler["M"])],
    "Min":  [np.min(wave_data["eta"]), np.min(force_wheeler["F"]), np.min(force_wheeler["M"])],
}
table = pd.DataFrame(stats, index=["Free Surface Elevation [m]", "Wheeler Force [N]", "Wheeler Moment [Nm]"])

print("\nTable with mean, standard deviation, min and max values\n")
print(table)

sigma_eta = table.loc["Free Surface Elevation [m]", "Std"]
Hs_from_calc = 4.0 * sigma_eta
Hs_from_given = wave_data.get("Hs", None)
print(f"Given significant wave height: {Hs_from_given} m; calculated Hs: {Hs_from_calc:.3f} m")

# Histogram for eta
plt.figure()
plt.hist(wave_data["eta"], bins=80)
plt.xlabel("η [m]")
plt.ylabel("Counts")
plt.title("Histogram of free-surface elevation η")
plt.grid(alpha=0.3)
plt.show()

print("Gamma used for JONSWAP:", wave_data["gamma"])