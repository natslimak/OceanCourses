# Exercise H01
# Programmed by David R. Fuhrman, Sept. 9, 2025
# Converted to Python

import numpy as np
from scipy.optimize import fsolve

# Input
Lab, Lbc, Dbc = 150, 200, 0.20
ks = 0.002
Qab, Qbc = 0.045, 0.085
zetaInlet, zetaOutlet, zetaElbow = 0.5, 1.0, 0.5
g, nu = 9.81, 1e-6
za, zb, zc = 32.56, 26.67, 90.0


def ColebrookWhiteAB(x, constants):
    g, nu, Q, z1, z2, zetaInlet, zetaOutlet, ks, L = constants
    f, D = x  # unknowns

    R = D / 4
    A = np.pi / 4 * D**2
    v = Q / A
    Re = R * v / nu

    F1 = z1 - z2 - (zetaInlet + zetaOutlet) * v**2 / (2 * g) - f * L / R * v**2 / (2 * g)
    F2 = np.sqrt(2 / f) - 6.4 + 2.45 * np.log(ks / R + 4.7 / (Re * np.sqrt(f)))
    return [F1, F2]


def ColebrookWhiteBC(x, constants):
    g, nu, Q, z1, z2, zetaInlet, zetaOutlet, zetaElbow, ks, L, D = constants
    f, Hpump = x  # unknowns

    R = D / 4
    A = np.pi / 4 * D**2
    v = Q / A
    Re = R * v / nu

    F1 = z1 + Hpump - z2 - (zetaInlet + zetaOutlet + 2 * zetaElbow) * v**2 / (2 * g) - f * L / R * v**2 / (2 * g)
    F2 = np.sqrt(2 / f) - 6.4 + 2.45 * np.log(ks / R + 4.7 / (Re * np.sqrt(f)))
    return [F1, F2]


# Reservoir A-B
constantsAB = (g, nu, Qab, za, zb, zetaInlet, zetaOutlet, ks, Lab)
guessAB = [0.01, 0.2]
solAB = fsolve(ColebrookWhiteAB, guessAB, args=(constantsAB,))
fab, Dab = solAB
print(f"fab = {fab:.6f}, Dab = {Dab:.6f}")

# Reservoir B-C
constantsBC = (g, nu, Qbc, zb, zc, zetaInlet, zetaOutlet, zetaElbow, ks, Lbc, Dbc)
guessBC = [0.01, 50]
solBC = fsolve(ColebrookWhiteBC, guessBC, args=(constantsBC,))
fbc, Hpump = solBC
print(f"fbc = {fbc:.6f}, Hpump = {Hpump:.6f}")
