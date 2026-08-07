""" Hydrodynamic Forces on a Monopile in Regular Waves with Wheeler-stretched Depths

    Scenario 1: Forces without vertical acceleration (drag only)
    Scenario 2: Forces with vertical acceleration (drag + added mass) """

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON
from waves import calculateRegularWaveParameters, calculateFreeSurfaceElevationTimeSeries, calculateKinematics
from monopile import forceIntegrate


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)

# Load wave and time data (same inputs as used in main_q2)
wave_data = loadFromJSON(get_input_file("wave2.json"))
time_data = loadFromJSON(get_input_file("time.json"))
wave_data.update(time_data)
wave_data["t"] = np.arange(0., wave_data["TDur"], wave_data["dt"])

# Recompute kinematics to ensure fields exist locally
wave_data = calculateRegularWaveParameters(wave_data)
wave_data = calculateFreeSurfaceElevationTimeSeries(wave_data)
wave_data = calculateKinematics(wave_data)

# Load monopile properties
monopile_props = loadFromJSON(get_input_file("monopile.json"))


# ============================================================================
# Compute Wheeler-stretched physical depths
# z_phys(t, z) = z + eta(t) * (1 + z/h)
# where z is the reference depth array (negative values) and h is water depth
# ============================================================================
z_ref = wave_data["z"]  # shape (nz,)
eta = wave_data["eta"]  # shape (nt,)
h = wave_data["h"]

# z_phys will have shape (nt, nz)
z_phys = z_ref[None, :] + eta[:, None] * (1.0 + z_ref[None, :] / h)
wave_data["z_phys"] = z_phys


# ============================================================================
# Force calculation helpers and storage
# We'll compute forces for two cases: nominal (use z_ref) and wheeler-stretched (use z_phys)
# For each case we compute: combined (u + ut), drag-only (u only), inertia-only (ut only)
# ============================================================================
nt = wave_data["t"].shape[0]
force_nominal = {"t": wave_data["t"], "F_combined": np.zeros(nt), "F_drag": np.zeros(nt), "F_inertia": np.zeros(nt)}
force_wheeler = {"t": wave_data["t"], "F_combined": np.zeros(nt), "F_drag": np.zeros(nt), "F_inertia": np.zeros(nt)}


# Compute forces using the provided monopile forceIntegrate function
for ti in range(nt):
    # common fields at this timestep
    u_t = wave_data["u"][ti, :]
    ut_t = wave_data["ut"][ti, :]

    # --- Nominal (reference depths) ---
    Fc, Mc = forceIntegrate(monopile_props, u_t, ut_t, z_ref, 0.0)
    Fd, Md = forceIntegrate(monopile_props, u_t, np.zeros_like(ut_t), z_ref, 0.0)
    Fi, Mi = forceIntegrate(monopile_props, np.zeros_like(u_t), ut_t, z_ref, 0.0)
    force_nominal["F_combined"][ti] = Fc
    force_nominal["F_drag"][ti] = Fd
    force_nominal["F_inertia"][ti] = Fi

    # --- Wheeler-stretched physical depths ---
    zphys_t = z_phys[ti, :]
    Fcw, Mcw = forceIntegrate(monopile_props, u_t, ut_t, zphys_t, 0.0)
    Fdw, Mdw = forceIntegrate(monopile_props, u_t, np.zeros_like(ut_t), zphys_t, 0.0)
    Fiw, Miw = forceIntegrate(monopile_props, np.zeros_like(u_t), ut_t, zphys_t, 0.0)
    force_wheeler["F_combined"][ti] = Fcw
    force_wheeler["F_drag"][ti] = Fdw
    force_wheeler["F_inertia"][ti] = Fiw


# ============================================================================
# Plot comparisons: combined, drag-only, inertia-only
# ============================================================================
def plot_comparison(time, series_a, series_b, title, label_a="Nominal", label_b="Wheeler", ylabel="Force [N]"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, series_a, label=label_a, linewidth=2)
    ax.plot(time, series_b, label=label_b, linewidth=2, linestyle='--')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


plot_comparison(force_nominal["t"], force_nominal["F_combined"], force_wheeler["F_combined"],
                "Total Force: Nominal vs Wheeler-stretched")

plot_comparison(force_nominal["t"], force_nominal["F_drag"], force_wheeler["F_drag"],
                "Drag-only Force: Nominal vs Wheeler-stretched")

plot_comparison(force_nominal["t"], force_nominal["F_inertia"], force_wheeler["F_inertia"],
                "Inertia-only Force: Nominal vs Wheeler-stretched")


