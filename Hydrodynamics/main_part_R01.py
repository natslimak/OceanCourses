'''
ASSIGNMENT 1
Wave perpendicular to the shoreline
'''

import numpy as np
import constants as const
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from waves import *

beta = 1/200                    # beach slope
d = 0.2 * 10**-3                # grain size [m]
a_0 = 0                         # angle of the approaching wave [degrees]
g = 9.81                        # gravity [m/s^2]
x = 300 *10**3                  # distance from the shoreline [m]
rho = 1025                      # water density [kg/m^3]

B = 10                          # Beaufort scale
u_10 = 0.836 * B**(3/2)         # wind speed at 10 m height [m/s]


# --------------------------------------------
# TASK 1 - Fully developed sea
# --------------------------------------------

# Check require wind duration
t_xu = 77.23 * x**0.67 / (u_10**0.34*g**0.33)                      # time to reach full development [s]

# Estimate friction velocity
C_d = 0.001 * (1.1 + 0.035 * u_10)                                 # drag coefficient
u_star = np.sqrt(C_d) * u_10                                       # friction velocity [m/s]

# Estimate wave height
H_m0 = 0.0413 * (g*x/u_star**2)**(1/2) * (u_star**2/g)             # significant wave height [m]
T_p = 0.751 * (g*x/u_star**2)**(1/3) * (u_star/g)                  # peak period [s]

H_m0_upper = 211.5 * (u_star**2/g)                                  # upper limit of significant wave height [m]
T_p_upper = 239.8 * (u_star/g)                                      # upper limit of peak period [s]

L_0 = g * T_p**2 / (2 * np.pi)                                      # deep water wave length [m]
k_0 = 2 * np.pi / L_0                                               # deep water wave number [m^-1]
c_0 = np.sqrt(g / k_0)                                              # deep water wave celerity [m/s]

# Print the results
print(f't_xu = {t_xu/3600:.2f} hours')
print(f'u_* = {u_star:.2f} m/s')
print(f'H_m0 = {H_m0:.2f} m')
print(f'T_p = {T_p:.2f} s')
print(f'Upper limit: H_m0 = {H_m0_upper:.2f} m')
print(f'Upper limit: T_p = {T_p_upper:.2f} s')

print('\n--- TASK 1 ---')
print(f'Deep water wave length L_0: {L_0:.2f} m')
print(f'Deep water wave number k_0: {k_0:.4f} m^-1')
print(f'Deep water wave celerity c_0: {c_0:.2f} m/s')


# --------------------------------------------
# TASK 2 - Breaking point
# --------------------------------------------

H_0 = H_m0 / 1.4               # convert H_m0 to H_0
KB = 0.8                       # coefficient for random wave breaking
T = T_p                        # wave period
omega = 2*np.pi/T              # angular frequency
k_0 = omega**2/g               # wave number of the deep water


# Plot H(D) for D = 2 - 100 --> not starting from 0 because it gives problems
Ds = np.linspace(2, 100, 600)              #D --> array of D from 2 to 100 m depth
Hs = np.zeros_like(Ds)                     #H --> array of the same dimension where I save H(D)
ks = np.zeros_like(Ds)                     #k --> array of the same dimension where I save H(D)

# Iterate over depths and calculate H(D)
for i, D in enumerate(Ds):
    Hs[i], ks[i] = shoaling(D, H_0, k_0, T)

# Calculation of the parameters at breaking point
Db, Hb, kb, cb = breaking_point(Ds, H_0, k_0, T, KB=0.8, tol=1e-2)                  # Using the given ratio from the assignment K = 0.8

# Plot the shoaling curve
mask_shoal = Ds >= Db                     # keep depths from offshore down to breaking depth
Ds_shoal = Ds[mask_shoal]
Hs_shoal = Hs[mask_shoal]
plt.plot(Ds_shoal, Hs_shoal, label="Shoaling curve")
plt.plot(Db, Hb, 'ro', label=f'Breaking point (Db={Db:.2f} m, Hb={Hb:.2f} m)')
plt.xlabel(r"Depth $D$ [m]")
plt.ylabel(r"Wave height $H$ [m]")
plt.title(r"Shoaling")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # shoreline on the right
plt.show()


# Print the results
print('\n--- TASK 2 ---')
print(f"Breaking depth ≈ {Db:.3f} m")
print(f"Breaking wave height ≈ {Hb:.3f} m")
print(f"Breaking wave number ≈ {kb:.6f} 1/m")
print(f"Breaking wave celerity ≈ {cb:.3f} m/s")


# --------------------------------------------
# TASK 3 - Wave height H(D)
# --------------------------------------------

D_start = 110.0    
D_end = 0.1      

# Create arrays for D, H(D) and k(D)
Ds = np.linspace(D_start, D_end, 600)
Hs = np.empty_like(Ds)
ks = np.empty_like(Ds)

# H(D) calculation -> we use shoaling before breaking, then linear relation for surf zone
for i, D in enumerate(Ds):
    if D >= Db:
        Hs[i], ks[i] = shoaling(D, H_0, k_0, T)

    else:
        Hs[i] = KB * D

# Plot the evolution of wave height H
plt.figure(alpha=0.5)
mask_out = Ds >= Db
plt.plot(Ds[mask_out], Hs[mask_out], label="Before breaking (shoaling)")
mask_in = Ds < Db
plt.plot(Ds[mask_in], Hs[mask_in], '--', label=f"Surf zone (H = {KB}·D)")
plt.axvline(Db, color='k', linestyle=':', label=f"Depth at breaking ≈ {Db:.2f} m")
plt.xlabel(r"Depth $D$ [m]")
plt.ylabel(r"Wave height $H$ [m]")
plt.title(r"Variation of the wave height $H$ with depth $D$")
plt.grid(True)
plt.gca().invert_xaxis()  # shoreline on the right
plt.legend()
plt.show()

# Print the results
print('\n--- TASK 3 ---')
print("The plot showing the Variation of the wave height H with depth D has been generated.")


# --------------------------------------------
# TASK 4 - Mean water level eta
# --------------------------------------------

eta_s = np.empty_like(Ds)                                       # mean water level array
G = np.empty_like(Ds)                                           # energy flux factor array

# Iterate over depths and calculate eta_s(D)
for i, D in enumerate(Ds):
    if D >= Db:
        G[i] = (2.0 * ks[i] * D) / np.sinh(2.0 * ks[i] * D)     # ks[i] and Hs[i] already calculated
        eta_s[i] = - (Hs[i]**2) * G[i] / (16.0 * D)             # wave set down
    else:
        eta_s[i] = - KB**2*Db/16 + 3/8*KB**2*(Db-D)             # wave set up using the approximated formula for shallow water

# Plot the results
plt.figure(alpha=0.5)
mask_out = Ds >= Db
plt.plot(Ds[mask_out], eta_s[mask_out], label=f"Before breaking (shoaling)")
mask_in = Ds < Db
plt.plot(Ds[mask_in], eta_s[mask_in], '--', label=f"Surf zone (H = {KB}·D)")
plt.axvline(Db, color='k', linestyle=':', label=f"Depth at breaking ≈ {Db:.2f} m")
plt.xlabel(r"Depth $D$ [m]")
plt.ylabel(r"Mean water level $\eta$ [m]")
plt.title(r"Variation of the mean water level $\eta$ with depth $D$")
plt.grid(True)
plt.gca().invert_xaxis()  # shoreline on the right
plt.legend()
plt.show()

# Print the results
print('\n--- TASK 4 ---')
print("The plot showing the Variation of the mean water level eta with depth D has been generated.")