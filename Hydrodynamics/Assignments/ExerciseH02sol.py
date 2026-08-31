# 41111-Hydrodynamics 2 Exercise 2
# Converted from MATLAB to Python
# David R. Fuhrman, Sept. 9, 2025

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# General Parameters
g = 9.81       # (m/s^2) Gravitational acceleration
nu = 1e-6      # (m^2/s) Kinematic viscosity of water

# Problem setup
q = 3.0        # (m^2/s) Discharge (per unit width)
d = 0.001      # (m) Diameter of sediment grain
k_s = 2.5 * d  # Nikuradse's equivalent sand grain roughness


def Q1(x, constants):
    g, nu, q, k_s, S = constants
    D, f, u = x  # unknowns: depth, friction factor, velocity

    # Equations
    F1 = u - np.sqrt(2 * g * D * S / f)  # flow resistance
    F2 = np.sqrt(2 / f) - (6.4 - 2.45 * np.log(k_s / D + 4.7 / (q / nu * np.sqrt(f))))
    F3 = u - q / D  # velocity-discharge relation
    return [F1, F2, F3]


# --- Question 1_1 ---
S = 0.001
constants = (g, nu, q, k_s, S)
initial = [1, 0.001, 3]
solution1 = fsolve(Q1, initial, args=(constants,))

D1_1, f1_1, u1_1 = solution1
F1_1 = u1_1 / np.sqrt(g * D1_1)

# --- Question 1_2 ---
S = 0.01
constants = (g, nu, q, k_s, S)
initial = [1, 0.001, 3]
solution2 = fsolve(Q1, initial, args=(constants,))

D1_2, f1_2, u1_2 = solution2
F1_2 = u1_2 / np.sqrt(g * D1_2)

print(f"Q1_1: D={D1_1:.6f}, f={f1_1:.6f}, u={u1_1:.6f}, Froude={F1_1:.6f}")
print(f"Q1_2: D={D1_2:.6f}, f={f1_2:.6f}, u={u1_2:.6f}, Froude={F1_2:.6f}")

# --- Question 2 ---
L = 10.0   # (m) Wavelength of the bottom undulation
b = 0.1    # (m) Amplitude of the bottom undulation

x = np.arange(0, 2 * L + 0.1, 0.1)  # x-coordinates

h = b * np.sin(2 * np.pi * x / L)  # wavy bed
eta_sub = b * np.sin(2 * np.pi * x / L) * (F1_1**2 / (F1_1**2 - 1))
eta_super = b * np.sin(2 * np.pi * x / L) * (F1_2**2 / (F1_2**2 - 1))

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x, h, label=r"$h$")
plt.plot(x, eta_sub + D1_1, label=r"$\eta_{subcritical}+D_{subcritical}$")
plt.plot(x, eta_super + D1_2, label=r"$\eta_{supercritical}+D_{supercritical}$")

plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.legend(loc="upper right")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.show()
