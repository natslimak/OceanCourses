"""" Bay Shape"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data 
wave_file = Path(__file__).parent / "Data" / "Hs_Tp_3h_GR8.xls"
df_wave = pd.read_excel(wave_file, sheet_name="Sheet1", header=None, names=["time", "Hs", "Tp"],)

drift_file = Path(__file__).parent / "Data" / "Drift_calculations_results.xlsx"
drift_sheet = pd.read_excel(drift_file, sheet_name=0, header=None)

# Import from the previous script
from A1_shoreline_properties import (null_net_orientation, d_l)


# =================================================================
# 1.1 New Coastline Orientation
# =================================================================

# Given parameters
R0 = 990                                    # length of the beach [m]
beach_norm = 292                            # orientation [degrees]
h_p = 1.5                                   # active beach height [m]
beta = beach_norm - null_net_orientation    # difference between orientation and null net orientation

C0 = -0.0002 * beta**2 + 0.0079 * beta - 0.0209
C1 = 0.0002 * beta**2 + 0.0004 * beta + 0.9437
C2 = -0.00005 * beta**2 - 0.0093 * beta + 0.0879

theta = np.linspace(beta, beta+90, 1000)           # range of orientations to evaluate
Rs = []

for theta_i in theta:
    R = R0 * (C0 + C1 * (beta / theta_i) + C2 * (beta / theta_i)**2)
    Rs.append(R)

Rs = np.asarray(Rs)

# Convert the polar coordinates to Cartesian coordinates.
# theta is in degrees for the engineering equations, so convert it to radians
# before using it with sin and cos.
theta_rad = np.deg2rad(theta)
x = Rs * np.cos(theta_rad)
y = Rs * np.sin(theta_rad)

# Plot the bay shape in a local Cartesian coordinate system.
fig, ax = plt.subplots()
ax.plot(x, y, color="tab:blue", linewidth=2, label="Shoreline")
ax.scatter(0, 0, color="black", zorder=3, label="Upcoast fixed point")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Parabolic bay shape")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
