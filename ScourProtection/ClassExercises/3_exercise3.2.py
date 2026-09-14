""" Determine if the difraction will occur """

import numpy as np
from scipy.optimize import fsolve

D = 70          # Monopile diameter [m]
h = 104         # Water depth [m]
H = 20          # Wave height [m]
T = 15          # Wave period [s]
g = 9.81        # Gravitational acceleration [m/s^2]


# Get the initial guess for the wavenumber
L0 = g * T**2 / (2 * np.pi)  # Initial guess for wavelength in deep water
k0 = 2 * np.pi / L0          # Initial guess for wavenumber

# Solve dispersion relation
func = lambda k0: g * k * np.tanh(k * h) - (2 * np.pi / T)**2
k = fsolve(func, k0)
k = k[0]

L = 2 * np.pi / k      # Wavelength

# Calculate the diffraction parameter
diff_ratio = D/L

# Calcualte the Ursell number
Ur = H * L**2 / (h**3)
if Ur < 15:
    print("Linear wave theory is valid")
else: 
    print("Non-linear wave theory is valid")

# Calulate the free-stream velocity
U_m = (np.pi * H / T) * (1 / np.sinh(k * h))

# Calculate the KC number 
KC = U_m * T / D


# Get the results    
print(f"Wave Length: {L:.2f}")
print(f"Diffraction ratio D/L: {diff_ratio:.2f}")
print(f"Ursell number: {Ur:.2f}")
print(f"Free-stream velocity: {U_m:.2f}")
print(f"KC number: {KC:.2f}")