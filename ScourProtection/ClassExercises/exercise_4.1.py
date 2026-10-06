"""  Estimate the KC_tot and U_m,top """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import numpy as np
from scipy.optimize import fsolve

# Inputs
D = 8.4  # Monopile diameter
h = 30  # Water depth far field
V = 0.5  # Depth averaged velocity
Hs = 8  # Significant wave height
Tp = 14  # Peak period
Tarmour = 0.91  # Thickness armour layer
g = 9.81  # Gravity

htop = h - Tarmour  # Water depth on top armour layer

# Calculate free-stream velocity magnitude using the Roulund et al. (2016) approach
Tz = Tp / 1.3
Um = (Hs / (2 * np.sqrt(2))) * np.sqrt(g / h) * np.exp(-(3.65 / Tz * np.sqrt(h / g)) ** 2.1)

# Calculate total KC number
KCw = Um * Tp / D  # Wave part of the KC number
KCcur = V * Tp / D  # Current part of the KC number
KCtot = KCcur + KCw  # Total KC number

# Amplification top of stones
L0 = g * Tp ** 2 / (2 * np.pi)  # Deep water wavelength
k0 = 2 * np.pi / L0  # Corresponding wave number

# Define the function for fsolve
def wave_number(k):
    return g * k * np.tanh(k * h) - (2 * np.pi / Tp) ** 2

# Solve for the actual wave number
k = fsolve(wave_number, k0)[0]

# Amplification factor
Ktop = np.sinh(k * h) / np.sinh(k * htop)

# Free-stream velocity magnitude on top of protection
Umtop = Um * Ktop