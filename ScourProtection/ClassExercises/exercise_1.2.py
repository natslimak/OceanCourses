import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# %% Inputs
V = 0.45          # depth averaged velocity
h = 0.5           # water depth
nu = 1e-6         # kinematic viscosity
kappa = 0.4       # von Karman constant

d = 3e-4          # grain size
ks = 2.5 * d      # Nikuradses roughness
g = 9.81          # gravitational acceleration
s = 2.65          # relative density

# %% Apply rough flow resistance formulation and calculate
# Shields parameter and grain Reynolds number

Uf = V / (6 + (1 / kappa) * np.log(h / ks))  # rough friction velocity

theta = Uf**2 / (g * (s - 1) * d)            # Shields parameter

Reg = Uf * d / nu                             # grain Reynolds number

print(f"Uf = {Uf:.6g}")
print(f"theta = {theta:.6g}")
print(f"Reg = {Reg:.6g}")

# %% Estimate critical Shields parameter

# Critical Shields equation
def critical_shields_equation(UfCr):
    Re = UfCr * d / nu

    return (
        0.165 * (Re + 0.6)**(-0.8)
        + 0.045 * np.exp(-40 * Re**(-1.3))
        - UfCr**2 / (g * (s - 1) * d)
    )

# Numerical solution
Uf_cr = brentq(
    critical_shields_equation,
    1e-8,
    1.2
)

theta_cr = Uf_cr**2 / (g * (s - 1) * d)

print(f"Uf_cr = {Uf_cr:.6g}")
print(f"theta_cr = {theta_cr:.6g}")

# %% Plot the Shields diagram over range of Uf values

Ufvec = np.arange(0.001, 1.201, 0.001)

# Grain Reynolds number
Re_vec = Ufvec * d / nu

# Critical Shields curve
theta_crit = (
    0.165 * (Re_vec + 0.6)**(-0.8)
    + 0.045 * np.exp(-40 * Re_vec**(-1.3))
)

# %% Plot

plt.figure()

plt.loglog(
    Re_vec,
    theta_crit,
    'k',
    label='Critical Shields curve'
)

plt.plot(
    Reg,
    theta,
    'kd',
    label='Present case'
)

plt.plot(
    Uf_cr * d / nu,
    theta_cr,
    'rx',
    label='Critical Shields of present case'
)

plt.xlabel(r'$\frac{U_f d}{\nu}$')
plt.ylabel(r'$\theta$')

plt.legend()
plt.grid(True, which='both', alpha=0.3)

plt.show()
# %%
