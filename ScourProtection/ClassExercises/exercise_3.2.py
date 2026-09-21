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


"""
import numpy as np
from scipy.optimize import fsolve

# Input parameters
D = 70  # Diameter (m)
T = 15  # Period (s)
h = 104  # Water depth (m)
g = 9.81  # Gravitational acceleration (m/s^2)
H = 20  # Wave height (m)

# Calculations
L0 = g * T**2 / (2 * np.pi)  # Deep water wavelength (m)
k0 = 2 * np.pi / L0  # Corresponding wave number (1/m)

# Define the function to solve for the actual wave number k
def wave_number_eq(k):
    return g * k * np.tanh(k * h) - (2 * np.pi / T)**2

# Solve for the actual wave number k
k = fsolve(wave_number_eq, k0)[0]

L = 2 * np.pi / k  # Calculate the wavelength (m)
DL = D / L  # Calculate the diameter to wavelength ratio
Urs = H * L**2 / h**3  # Calculate the Ursell number
Um = np.pi * H / T * 1 / np.sinh(k * h)  # Estimate the free-stream velocity magnitude
KC = Um * T / D  # Calculate the KC number
KCBreak = 0.44 / DL  # KC number inducing breaking

# Display results
print(f"Wavelength L: {L:.2f} m")
print(f"Diameter to wavelength ratio DL: {DL:.4f}")
print(f"Ursell number Urs: {Urs:.4f}")
print(f"Free-stream velocity magnitude Um: {Um:.4f} m/s")
print(f"KC number: {KC:.4f}")
print(f"KC number inducing breaking KCBreak: {KCBreak:.4f}")
"""