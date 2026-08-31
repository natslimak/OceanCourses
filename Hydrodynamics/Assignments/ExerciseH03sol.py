import sympy as sp
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ===========================================
# Question 1
# ===========================================

# Define symbolic variables
c1, c2, b1, h0, V, g, k, D, Fr, kD = sp.symbols('c1 c2 b1 h0 V g k D Fr kD')

# Define equations
eq1 = c1 - c2 - (h0 * V)                # Eq 3.47
eq2 = c1 * sp.exp(k * D) - c2 * sp.exp(-k * D) - (V * b1)   # Eq 3.51
eq3 = c1 * sp.exp(k * D) + c2 * sp.exp(-k * D) - (g * b1 / (V * k))   # Eq 3.57

# Solve 3 equations for 3 unknowns
S = sp.solve([eq1, eq2, eq3], (c1, c2, b1), dict=True)[0]

c1_sol = sp.simplify(S[c1])
c2_sol = sp.simplify(S[c2])
b1_sol = sp.simplify(S[b1])

# Change exponential terms to sinh/cosh
c1_sol = c1_sol.rewrite(sp.sinh)
c2_sol = c2_sol.rewrite(sp.sinh)
b1_sol = b1_sol.rewrite(sp.sinh)

# Define Froude number
eq4 = sp.Eq(Fr, V / sp.sqrt(g * D))
V_sol = sp.solve(eq4, V)[0]

# b1/h0
b1h0 = sp.simplify(b1_sol / h0)
b1h0 = b1h0.rewrite(sp.sinh)

# Check against eq 3.58
eq5 = (sp.cosh(k * D) - sp.sinh(k * D) / (k * D * Fr**2))**(-1)
Q1_check = sp.simplify(b1h0 - eq5)   # should simplify to 0

print("Check Q1 (should be 0):", Q1_check)


# ===========================================
# General Parameters
# ===========================================
g = 9.81     # (m/s^2)
nu = 1e-6    # (m^2/s)

# ===========================================
# Question 2
# ===========================================
h0 = 0.04   # mound height (m)
lam = 0.4   # mound width (m)
L = 2 * lam # bottom wavelength (m)
k = 2 * np.pi / L  # wavenumber (1/m)

x = np.arange(0, 2*L+0.01, 0.01)
h = h0 * np.sin(k * x)

# ---- Helper function for fsolve ----
def Q2(vars, constants):
    V1, V2 = vars
    g, D1, D2, h0 = constants
    F1 = D1 + V1**2 / (2*g) - ((D2 + h0) + V2**2 / (2*g))  # Bernoulli
    F2 = V1*D1 - V2*D2                                    # Continuity
    return [F1, F2]

# ---- Subcritical Flow ----
D1_sub, D2_sub = 0.11, 0.063
initial = [2, 2]
constants = [g, D1_sub, D2_sub, h0]
V1_sub, V2_sub = fsolve(Q2, initial, args=(constants,))

F1_sub = V1_sub / np.sqrt(g*D1_sub)
F2_sub = V2_sub / np.sqrt(g*D2_sub)
Fc_sub = np.sqrt(np.tanh(k*D1_sub)/(k*D1_sub))

b1_sub = h0 / (np.cosh(k*D1_sub) - np.sinh(k*D1_sub)/(k*D1_sub*F1_sub**2))
eta_sub = b1_sub/h0 * h

check1_sub = b1_sub/h0 - (F1_sub**2/(F1_sub**2 - 1))
check2_sub = (D1_sub + b1_sub - h0) - D2_sub

print("Subcritical checks:", check1_sub, check2_sub)

# ---- Supercritical Flow ----
D1_super, D2_super = 0.065, 0.067
constants = [g, D1_super, D2_super, h0]
V1_super, V2_super = fsolve(Q2, initial, args=(constants,))

F1_super = V1_super / np.sqrt(g*D1_super)
F2_super = V2_super / np.sqrt(g*D2_super)
Fc_super = np.sqrt(np.tanh(k*D1_super)/(k*D1_super))

b1_super = h0 / (np.cosh(k*D1_super) - np.sinh(k*D1_super)/(k*D1_super*F1_super**2))
eta_super = b1_super/h0 * h

check1_super = b1_super/h0 - (F1_super**2/(F1_super**2 - 1))
check2_super = (D1_super + b1_super - h0) - D2_super

print("Supercritical checks:", check1_super, check2_super)

# ===========================================
# Plot results
# ===========================================
plt.figure(figsize=(8,5))
plt.plot(x, h, label='$h$')
plt.plot(x, eta_sub + D1_sub, label='$\eta_{subcritical}+D_{subcritical}$')
plt.plot(x, eta_super + D2_super, label='$\eta_{supercritical}+D_{supercritical}$')
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
plt.ylim([-0.05, 0.15])
plt.legend()
plt.grid(True, which='both')
plt.show()