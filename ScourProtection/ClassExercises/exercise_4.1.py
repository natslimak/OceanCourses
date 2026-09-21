"""  Estimate the KC_tot and U_m,top """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

D = 8.4             # Diameter of the monopile [m]
Hs = 8.0            # Significant wave height [m]
Tp = 14.0           # Peak wave period [s]
hw = 20.0           # Water depth [m]
V = 0.5             # Current velocity [m/s]
t_rock = 0.91       # Rock size [m]
g = 9.81            # Acceleration due to gravity [m/s^2]
h_top = hw-t_rock   # Height of the scour protection [m]


# Zero-crossing wave period
Tz = Tp / 1.3

# Free stream velocity at the top of the pipeline
U_m = (Hs / (2 * np.sqrt(2))) * np.sqrt(g/hw) * np.exp(-((3.65/Tz) * np.sqrt(hw/g))**2.1)

# Total Keulegan-Carpenter numbers
KC_w = U_m * Tp / D
KC_c = V * Tp / D
Kc_tot = KC_w + KC_c

# Get the initial guess for the wavenumber
L0 = g * Tp**2 / (2 * np.pi)  # Initial guess for wavelength in deep water
k0 = 2 * np.pi / L0           # Initial guess for wavenumber

# Solve dispersion relation
func = lambda k0: g * k * np.tanh(k * hw) - (2 * np.pi / Tp)**2
k = fsolve(func, k0)
k = k[0]

L = 2 * np.pi / k      # Wavelength

# Factor relating velocity at seabed level to velocity on scour protection
K_top = np.sinh(2*np.pi*hw/L) / (np.sinh(2*np.pi*(h_top)/L))
