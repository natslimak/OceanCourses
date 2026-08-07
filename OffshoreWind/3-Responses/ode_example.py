import os
import sys 
import numpy as np

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from integration import ode4

structure = {}
structure["GM"], structure["GD"], structure["GK"] = 1.0, 0.1, 1.0
D_crit = np.sqrt(2*structure["GK"]*structure["GM"])

# structure["GD"] = D_crit

forcing = {}
forcing["F0"],  forcing["Fa"], forcing["omega_f"] = 0., 1.0, 0.1

q0 = np.array([1.0, 0.0]) # (alpha, alphaDot) #Initial conditions
tspan = np.arange(0., 200., 0.1) # time vector for integration

def dqdt(t, q, structure, forcing):
    # Initialize dqdt
    dqdt = np.zeros_like(q)
    # compute forcing
    GF = forcing["F0"] + forcing["Fa"]*np.sin(forcing["omega_f"]*t)
    # compute the dqdt
    dqdt[0] = q[1]
    dqdt[1] = (GF - structure["GD"]*q[1] - structure["GK"]*q[0]) / structure["GM"]
    
    return dqdt

# Runge-Kutta 4th order integration method
q = ode4(dqdt, tspan, q0, structure, forcing)

import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot(tspan, q[:,0])
plt.show()