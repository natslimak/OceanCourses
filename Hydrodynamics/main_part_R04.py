'''
ASSIGNMENT 4
Turbulent Wave Boundary Layer
'''

import numpy as np
import constants as const
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from waves import *
from main_part_R01 import Db, Hb, kb, T  # import breaking depth, height, wavenumber and period

beta = 1/200                    # beach slope
d = 0.2 * 10**-3                # grain size [m]
a_0 = 0                         # angle of the approaching wave [degrees]
g = 9.81                        # gravity [m/s^2]
x = 300 *10**3                  # distance from the shoreline [m]
rho = 1000                      # water density [kg/m^3]
rho_air = 1.2                   # air density [kg/m^3]

B = 10                          # Beaufort scale
u_10 = 0.836 * B**(3/2)         # wind speed at 10 m height [m/s]
chi = 0.4
ks = 2.5*d


# --------------------------------------------------------
# TASK 1 - Free stream velocity profile
# --------------------------------------------------------

t = np.linspace(0.001*T, T/2, 600)

# Compute angular frequency
omega = 2*np.pi/T
y=0
U_m = (Hb*omega/2)*np.cosh(kb*y)/np.sinh(kb*Db)
# Print the results
print('\n--- TASK 1 ---')
print(f"Maximum free stream velocity U_m: {U_m:.4f} m/s")


# --------------------------------------------------------
# TASK 2 
# --------------------------------------------------------

# Domain and boundary conditions
t_0 = 0.001*T
t_fin = T/2
U_f_0 = 0.001*U_m  

def solve_U_f(t_0, t_fin, U_f_0, ks):
    
    def ode(t, U_f):
        
        Uf = U_f[0]
        
        U_0 = U_m * np.sin(omega * t)
        
        # delta(t)
        delta = ks/30 * np.exp(chi * U_0 / Uf)
        denom = (delta/chi) * (np.log(30*delta/ks) - 1)
        
        dUf_dt = (delta * U_m * omega * np.cos(omega * t) - Uf**2)/ denom
        
        return [dUf_dt]
    
    sol = solve_ivp(
        ode,
        (t_0, t_fin),
        [U_f_0],         
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        max_step=1000
    )
    
    ts = np.linspace(t_0, t_fin, 600)
    U_fs = sol.sol(ts)[0]   
    U_f = np.abs(U_fs)
    return ts, U_f, sol

ts, U_f, sol = solve_U_f(t_0, t_fin, U_f_0, ks)

# Plot temporal evolution of U_f
plt.figure(figsize=(10, 5))
plt.plot(ts, U_f, label='Friction velocity $U_f(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Friction velocity $U_f$ [m/s]')
plt.title('Time evolution of near-bed friction velocity $U_f(t)$')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

ts, U_f, sol = solve_U_f(t_0, t_fin, U_f_0, ks)
t_target = T/4
Uf_T4 = sol.sol(t_target)[0]
Uf_max = np.max(U_f)
U_0max = U_m
delta = ks/30 * np.exp(chi * U_0max / Uf_T4)

# Print the results
print('\n--- TASK 2 ---')
print(f"Uf max case 1: {Uf_max:.4f} m/s")
print(f"Max boundary layer thickness: {delta:.4f} m")



# --------------------------------------------------------
# TASK 3
# --------------------------------------------------------
a = U_m / omega
f_w = np.exp(5.5*(a/ks)**(-0.16)-6.7)

Uf_max_1 = np.sqrt(f_w/2)*U_m

#  Print the results
print('\n--- TASK 3 ---')
print(f"Uf max case 2: {Uf_max_1:.4f} m/s")


# --------------------------------------------------------
# TASK 4
# -------------------------------------------------------
s = 2.65
theta_max = Uf_max**2/((s-1)*g*d)

#  Print the results
print('\n--- TASK 4 ---')
print(f"Theta: {theta_max:.4f}")