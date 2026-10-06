import numpy as np
from scipy.optimize import fsolve

Hs = 1.4           # Wave height (m)
Tp = 2             # Wave period (s)
g = 9.81          # Acceleration due to gravity (m/s^2)
h = 8             # Water depth (m)
d = 0.28 * 1e-3   # Sediment diameter (m)
s = 2.65          # Relative density of sediment
nu = 1e-6         # Kinematic viscosity of water (m^2/s)
ks = 2.5 * d      # Roughness height (m)


L0 = g * Tp**2 / (2 * np.pi)    # Initial guess for wavelength in deep water

def dispersion_relation(L):
    return (g * Tp**2/(2*np.pi)) * np.tanh((2*np.pi*h)/L) - L

L = fsolve(dispersion_relation, L0)[0]
U_m = np.pi * Hs / Tp  # Maximum orbital velocity
# === Calculate the theta_cw === 
a = U_m * Tp / (2 * np.pi)  # Orbital wave amplitude

# Orbital wave amplitude
a = U_m * Tp / (2 * np.pi)

# Calculate the Reynolds number for the waves
Re_w = U_m * a / nu  

# Friction coefficient for the waves
f_w_lam = 2 / (np.sqrt(Re_w))                       # Laminar flow
f_w_smooth = 0.04 * Re_w ** (-0.16)                 # Smooth turbulent flow
f_w_rough = np.exp(5.5 * (a/ks) ** (-0.16) - 6.7)   # Rough turbulent flow

f_w = max(f_w_lam, f_w_smooth, f_w_rough)

# Calculate the friction velocity for the waves
U_fw = np.sqrt(f_w / 2) * U_m 

def solve_settling_velocity_and_drag(s, g, d, nu):
	"""Solve the coupled settling velocity and drag coefficient equations."""
	def equations(unknowns):
		w_s, C_D = unknowns
		return [
			w_s - np.sqrt(4 * (s - 1) * g * d / (3 * C_D)),
			C_D - (1.4 + 36 * nu / (w_s * d)),
		]

	initial_guess = [0.03, 5.0]
	solution, _, status, message = fsolve(
		equations, initial_guess, full_output=True
	)
	if status != 1:
		raise RuntimeError(f"Solver did not converge: {message}")
	return solution


# Settling velocity of sediment and drag coefficient
w_s, C_D = solve_settling_velocity_and_drag(s, g, d, nu)