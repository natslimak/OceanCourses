""" Hydrodynamic Forces on a Monopile in Regular Waves 

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

from waves import calculateRegularWaveParameters, calculateFreeSurfaceElevationTimeSeries, calculateKinematics
from common import loadFromJSON
from monopile import forceIntegrate


# ============================================================================
# SETUP: Load Input Files
# ============================================================================

# Define input file directory and helper function
input_dir = "inputVariables"
get_input_file = lambda filename: os.path.join(os.path.dirname(__file__), input_dir, filename)

# Load wave properties (period, amplitude, etc.)
wave_data = loadFromJSON(get_input_file("wave2.json"))

# Load time discretization settings (duration, time step)
time_data = loadFromJSON(get_input_file("time.json"))
wave_data.update(time_data)

# Create time vector from 0 to duration with specified time step
wave_data["t"] = np.arange(0., wave_data["TDur"], wave_data["dt"])

# ============================================================================
# WAVE KINEMATICS: Calculate wave properties and water motion
# ============================================================================

# Calculate wave parameters (wavelength, wavenumber, etc.)
wave_data = calculateRegularWaveParameters(wave_data)

# Calculate surface elevation (η) over time
wave_data = calculateFreeSurfaceElevationTimeSeries(wave_data)

# Calculate water particle velocities and accelerations at all depths
wave_data = calculateKinematics(wave_data)

# ============================================================================
# LOAD MONOPILE PROPERTIES
# ============================================================================

monopile_properties = loadFromJSON(get_input_file("monopile.json"))

# ============================================================================
# CALCULATE FORCES (Scenario 1: Without vertical acceleration)
# ============================================================================

# Initialize force arrays
force_data = {}
force_data["t"] = wave_data["t"]
force_data["horizontal_force"] = np.zeros_like(wave_data["t"])
force_data["moment"] = np.zeros_like(wave_data["t"])

# Calculate forces at each time step considering only horizontal velocity (drag forces)
for time_index, time_value in enumerate(wave_data["t"]):
    # Get horizontal velocity at this time step across all depths
    horizontal_velocity = wave_data["u"][time_index, :]
    
    # Zero out vertical acceleration for this scenario
    vertical_acceleration = np.zeros_like(wave_data["ut"][time_index, :])
    
    # Integrate forces over monopile depth
    force_data["horizontal_force"][time_index], force_data["moment"][time_index] = \
        forceIntegrate(monopile_properties, horizontal_velocity, vertical_acceleration, 
                      wave_data["z"], 0.)

# ============================================================================
# PLOT 1: Surface elevation and forces WITHOUT vertical acceleration
# ============================================================================

fig, axes = plt.subplots(2, figsize=(10, 8))

# Plot 1a: Surface elevation
axes[0].plot(wave_data["t"], wave_data["eta"], linewidth=2)
axes[0].set_title("Water Surface Elevation vs Time", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Surface Elevation η(t) [m]")
axes[0].grid(True, alpha=0.3)

# Plot 1b: Horizontal forces
axes[1].plot(force_data["t"], force_data["horizontal_force"], linewidth=2)
axes[1].set_title("Horizontal Force on Monopile vs Time (Drag Only)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Horizontal Force F(t) [N]")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# CALCULATE FORCES (Scenario 2: WITH vertical acceleration)
# ============================================================================

# Recalculate forces including vertical acceleration effects (more realistic model)
# This accounts for added mass effects from accelerating water
for time_index, time_value in enumerate(wave_data["t"]):
    # Get horizontal velocity at this time step
    horizontal_velocity = wave_data["u"][time_index, :]
    
    # Now include vertical acceleration (this increases forces at wave peaks/troughs)
    vertical_acceleration = wave_data["ut"][time_index, :]
    
    # Integrate forces over monopile depth
    force_data["horizontal_force"][time_index], force_data["moment"][time_index] = \
        forceIntegrate(monopile_properties, horizontal_velocity, vertical_acceleration, 
                      wave_data["z"], 0.)

# ============================================================================
# PLOT 2: Surface elevation and forces WITH vertical acceleration
# ============================================================================

fig, axes = plt.subplots(2, figsize=(10, 8))

# Plot 2a: Surface elevation
axes[0].plot(wave_data["t"], wave_data["eta"], linewidth=2)
axes[0].set_title("Water Surface Elevation vs Time", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Surface Elevation η(t) [m]")
axes[0].grid(True, alpha=0.3)

# Plot 2b: Horizontal forces (now with vertical acceleration effect)
axes[1].plot(force_data["t"], force_data["horizontal_force"], linewidth=2, color='orange')
axes[1].set_title("Horizontal Force on Monopile vs Time (Drag + Added Mass)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Horizontal Force F(t) [N]")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# PLOT 3: Surface elevation, velocity, and acceleration at depths
# ============================================================================

# Identify depth indices
surface_index = 34          # Water surface (z = 0 m)
seabed_index = 0            # Seabed (z = -34 m)

fig, axes = plt.subplots(3, figsize=(10, 10))

# Plot 3a: Surface elevation
axes[0].plot(wave_data["t"], wave_data["eta"], linewidth=2)
axes[0].set_title("Water Surface Elevation vs Time", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Surface Elevation η(t) [m]")
axes[0].grid(True, alpha=0.3)

# Plot 3b: Horizontal velocity at surface and seabed
axes[1].plot(wave_data["t"], wave_data["u"][:, seabed_index], label='At seabed (z = -34 m)', linewidth=2)
axes[1].plot(wave_data["t"], wave_data["u"][:, surface_index], label='At surface (z = 0 m)', linewidth=2)
axes[1].set_title("Horizontal Velocity at Different Water Depths", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity u(t) [m/s]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3c: Horizontal acceleration at surface and seabed
axes[2].plot(wave_data["t"], wave_data["ut"][:, seabed_index], label='At seabed (z = -34 m)', linewidth=2)
axes[2].plot(wave_data["t"], wave_data["ut"][:, surface_index], label='At surface (z = 0 m)', linewidth=2)
axes[2].set_title("Horizontal Acceleration at Different Water Depths", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Acceleration ut(t) [m/s²]")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()