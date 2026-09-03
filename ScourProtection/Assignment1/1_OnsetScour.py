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
# TASK 1a: Onset Scour - Steady Current
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
L_side_c = U **2 / g * D * (s - 1) * (1 - n)
R_side_c = lambda ratio: 0.025 * np.exp(9 * np.sqrt(ratio)) # ratio = e / D


# Plot the data
plt.figure(figsize=(8, 4), dpi=150)
plt.plot(ratios, R_side_c(ratios), color='red', linewidth=2)
plt.plot(e / D, L_side_c, 'o', label='Case 1', color='blue', markersize=7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.xlabel(r'$e/D$', fontsize=12)
plt.ylabel(r'$\frac{U^{2}}{gD(s-1)(1-n)}$', rotation=90, labelpad=16, fontsize=12)
plt.title('Onset Scour: Steady Current', fontsize=14)
plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# Get the results
print('Results for Onset Scour - Steady Current:')
print(f'L_side_c: {L_side_c:.4f}', '   ', f'R_side_c: {R_side_c(e / D):.4f}')
print('\nOnset scour occurs if L_side_c > R_side_c:')
print(f'Value of L_side / R_side: {L_side_c / R_side_c(e / D):.4f}')
print('Onset scour status: ', 'Yes' if L_side_c > R_side_c(e / D) else 'No')


# === Test of the other function === 
f = 0.025 * np.exp(9 * np.sqrt(e / D))
dp_dx = - rho * U ** 2 / (D * f)
onset_scour = np.abs(dp_dx) / (g * rho * (s-1) * (1-n))

print('\nOnset scour occursif pressure gradient / (g * rho * (s-1) * (1-n)) > 1:')
print(f'Onset scour value: {onset_scour:.2f}')   # His value: 0.65
print('Onset scour status: ', 'Yes' if onset_scour > 1 else 'No')


# ======================================================================
# TASK 1b: Onset Scour - Waves
# ======================================================================

# Get the initial guess for the wavenumber 
omega = 2 * np.pi / Tp        # (1/s) Wave angular frequency
k0 = omega**2 / g             # Initial guess for wavenumber

# Solve dispersion relation using the initial guess
func = lambda k0: omega**2 - g * k0 * np.tanh(k0 * D)
k = fsolve(func, k0)
k = k[0]

# Calculate the wavelength
L = 2 * np.pi / k     

# Zero-crossing wave period
Tz = Tp / 1.3

# Free stream velocity at the top of the pipeline
U_m = Hs / 2 * np.sqrt(2) * np.sqrt(g/h) * np.exp(-(3.65/Tz * np.sqrt(g/h))**2.1)

# Calculate the onset criteria
L_side_w = U_m **2 / g * D * (s - 1) * (1 - n)
R_side_w = lambda ratio: 0.025 * np.exp(9 * np.sqrt(ratio)) # ratio = e / D

# Get the KC number
KC = U_m * Tp / D
func_KC = lambda D: U_m * Tp / D

# Plot the data
plt.figure(figsize=(8, 4), dpi=150)
plt.plot(ratios, R_side_w(ratios), label=r'$R_{side}$', color='red', linewidth=2)
plt.plot(e / D, L_side_w, 'o', label='Onset Scour', color='blue', markersize=7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.xlabel(r'$e/D$', fontsize=12)
plt.ylabel(r'$\frac{U_{m}^{2}}{gD(s-1)(1-n)}$', rotation=90, labelpad=16, fontsize=12)
plt.title('Onset Scour: Waves', fontsize=14)
plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# Get the results
print('Results for Onset Scour - Waves:')
print(f'L_side_w: {L_side_w:.4f}', '   ', f'R_side_w: {R_side_w(e / D):.4f}')
print('\nOnset scour occursif L_side_w > R_side_w:')
print(f'Value of L_side / R_side: {L_side_w / R_side_w(e / D):.4f}')
print('Onset scour status: ', 'Yes' if L_side_w > R_side_w(e / D) else 'No')



# ======================================================================
# Plotting 
# ======================================================================

# Plot together on one plot both curves
plt.figure(figsize=(8, 4), dpi=150)
# Steady current
plt.plot(ratios, R_side_c(ratios), label=r'$R_{side}$ (Steady Current)', color='red', linewidth=2)
plt.plot(e / D, L_side_c, 'o', label='Onset Scour (Steady Current)', color='blue', markersize=7)
# Waves
plt.plot(ratios, R_side_w(ratios), label=r'$R_{side}$ (Waves)', color='orange', linewidth=2)
plt.plot(e / D, L_side_w, 'o', label='Onset Scour (Waves)', color='green', markersize=7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.xlabel(r'$e/D$', fontsize=12)
plt.ylabel(r'$\frac{U^{2}}{gD(s-1)(1-n)}$', rotation=90, labelpad=16, fontsize=12)
plt.title('Onset Scour: Steady Current vs Waves', fontsize=14)
plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

