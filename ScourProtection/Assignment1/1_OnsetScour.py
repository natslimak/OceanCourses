import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


# Constants
rho = 1000          # Water density [kg/m^3]
rho_s = 2650        # Sediment density [kg/m^3]
g = 9.81            # Gravitational acceleration [m/s^2]
kappa = 0.4         # von Karman constant [-]

# Given parameters
n = 0.43            # Porosity [-]
d50 = 0.2e-3        # Median grain size [m]
D = 0.8             # Diameter of the pipeline [m]
e = 0.04            # Initial Burial Depth [m]
h = 15.0            # Water depth [m]
ks = 2.5 * d50      # Roughness height [m]
s = 2.65            # Relative density of sediment [-]

# Spring tide current velocity properties
V = 1.1             # Depth-averaged current velocity [m/s]

# Dominant wave properties
Tp = 9.0            # Peak wave period [s]
Hs = 4.0            # Significant wave height [m]




# ======================================================================
# TASK 1: Onset Scour 
# ======================================================================

# Top of the pipeline (from the bottom)
z = D - e

# Define a range of burial ratios to plot the onset criteria as curves
ratios = np.geomspace(1e-3, 1, 200)

# Get the friction velocity
U_fc = V / (6 + (1/kappa) * np.log(h/ks))

# Get the velocity at the given depth
U = U_fc / kappa * np.log(30 * z / ks)

# Calculate the onset criteria
L_side = U / g * D * (s - 1) * (1 - n)
R_side = lambda ratio: 0.025 * np.exp(9 * np.sqrt(ratio)) # ratio = e / D


# Plot the data
plt.figure(figsize=(8, 4), dpi=150)
plt.plot(ratios, R_side(ratios), label=r'$R_{side}$', color='red', linewidth=2)
plt.plot(e / D, L_side, 'o', label='Onset Scour', color='blue', markersize=7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-2, 1)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.xlabel(r'$e/D$', fontsize=12)
plt.ylabel(r'$\frac{U^{2}}{gD(s-1)(1-n)}$', rotation=90, labelpad=16, fontsize=12)
plt.title('Onset Scour', fontsize=14)
plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# Get the results
print(f'L_side: {L_side:.4f}', '   ', f'R_side: {R_side(e / D):.4f}')
print('\nOnset scour occursif L_side > R_side:')
print('Onset scour status: ', 'Yes' if L_side > R_side(e / D) else 'No')


# Their value: 0.65
