'''
ASSIGNMENT 2
Waves attacking at an angle
'''

import numpy as np
import constants as const
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from waves import *
from scipy import integrate

beta = 1/200                          # beach slope
d = 0.2 * 10**-3                      # grain size [m]
alpha_0_deg = 20                      # angle of the approaching wave [degrees]
alpha_0 = np.deg2rad(alpha_0_deg)     #alpha in rad
g = 9.81                              # gravity [m/s^2]
x = 300 *10**3                        # distance from the shoreline [m]
rho = 1025                            # water density [kg/m^3]

B = 10                          # Beaufort scale
u_10 = 0.836 * B**(3/2)         # wind speed at 10 m height [m/s]

# Same equations from question R01
C_d = 0.001 * (1.1 + 0.035 * u_10)             
u_star = np.sqrt(C_d) * u_10
H_m0 = 0.0413 * (g*x/u_star**2)**(1/2) * (u_star**2/g)  
T_p = 0.751 * (g*x/u_star**2)**(1/3) * (u_star/g)
L_0 = g * T_p**2 / (2 * np.pi)                              # deep water wave length [m]
k_0 = 2 * np.pi / L_0                                       # deep water wave number [m^-1]
c_0 = np.sqrt(g / k_0)                                      # deep water wave celerity [m/s]


# --------------------------------------------
# TASK 1 - Wave number and angle at offshore
# --------------------------------------------

H_0 = H_m0 / 1.4               # convert H_m0 to H_0
KB = 0.8                       # coefficient for random wave breaking
T = T_p                        # peak period [s]
omega = 2*np.pi/T              # angular frequency [rad/s]  
k_0 = omega**2/g               # wave number of the deep water

# Calculation of the parameters at breaking point --> with refraction
Ds = np.linspace(2, 100, 600) 
Db, Hb, kb, cb, alphab = breaking_point_refraction(Ds, H_0, k_0, alpha_0, T, KB=0.8, tol=1e-2)
alphab_deg = np.rad2deg(alphab)

# Print the results
print('\n--- TASK 1 ---')
print(f"Wave breaking depth Db: {Db:.2f} m")
print(f"Wave breaking height Hb: {Hb:.2f} m")
print(f"Wave breaking wave number kb: {kb:.4f} m^-1")
print(f"Wave breaking celerity cb: {cb:.2f} m/s")
print(f"Wave breaking angle at breaking point alphab: {alphab_deg:.2f} degrees")


# --------------------------------------------
# TASK 2 - Wave height and angle evolution
# --------------------------------------------

# Plot the evolution of wave heigth H
D_start = 110.0    
D_end = 0.0

Ds = np.linspace(D_start, D_end, 600)
Hs = np.empty_like(Ds)
ks = np.empty_like(Ds)
alphas = np.empty_like(Ds)

# H(D) calculation -> we use refraction before breaking, then linear relation for surf zone
for i, D in enumerate(Ds):
    if D >= Db:
        Hs[i], ks[i], alphas[i] = refraction(D, H_0, k_0, alpha_0, T)
    else:
        Hs[i] = KB * D

# Plot the results        
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
print('\n--- TASK 2 ---')
print(f"The plot showing the Variation of the wave height H with depth D has been generated.")


# --------------------------------------------
# TASK 3 - Wave height and angle evolution
# --------------------------------------------

# Plot alpha
D_plot = np.linspace(3, 100, 600)
for i, D in enumerate(D_plot):
     Hs[i], ks[i], alphas[i] = refraction(D, H_0, k_0, alpha_0, T)

# Convert alphas to degrees for better readability
alphas_deg = np.rad2deg(alphas)

# Plot the results
plt.figure(alpha=0.5)
plt.plot(D_plot, alphas_deg)
plt.xlabel(r"Depth $D$ [m]")
plt.ylabel(r"Wave angle $\alpha$ [deg]")
plt.title(r"Variation of the wave angle $\alpha$ with depth $D$")
plt.grid(True)
plt.gca().invert_xaxis()  # shoreline on the right
plt.show()

# Print the results
print('\n--- TASK 3 ---')
print(f"The plot showing the Variation of the wave angle alpha with depth D has been generated.")


# --------------------------------------------
# TASK 4 - LONG SHORE CURRENT PROFILE
# --------------------------------------------
K_3 = 0.1                                   # coefficient for long-shore current

# Initialize arrays
D_start = Db                                # from the breaking depth
D_end = 0.0                                 # to the shoreline
Ds = np.linspace(D_start, D_end, 600)
U_c = np.empty_like(Ds)

# Velocity profile calculation for each depth D
for i, D in enumerate(Ds):
    U_c[i] = K_3 * D

# Plot the results
plt.figure(alpha=0.5)
plt.plot(Ds, U_c)
plt.xlabel(r"Depth $D$ [m]")
plt.ylabel(r"Long-shore current $U_{c}$ [m/s]")
plt.title(r"Variation of the long-shore current $U_{c}$ with depth $D$")
plt.grid(True)
plt.gca().invert_xaxis()  # shoreline on the right
plt.show()

# Calculate discharge
Q = integrate.quad(lambda D: K_3 * D, 0, Db)[0] * rho

# Print the results
print('\n--- TASK 4 ---')
print(f"Discharge Q of the long-shore current ≈ {Q:.3f} m^3/s")


# ------------------------------------------------------------
# TASK 5 - INSTANTANEOUS RATE OF LONGSHORE SEDIMENT TRANSPORT
# ------------------------------------------------------------

K_cerc = 0.77                                           # coefficient for sediment transport
n = 0.4                                                 # porosity of the sediment
rho_s = 2650                                            # sand density [kg/m^3]
p = 0.015                                               # occurance probability of the waves    

E_b = rho * g * Hb**2 / 8                               # wave energy at breaking point
P_l = E_b * cb * np.sin(alphab) * np.cos(alphab)        # long-shore wave power at breaking point

I_l = K_cerc * P_l                                      # instantaneous rate of longshore sediment transport [N/s]

Q_l = K_cerc * P_l / ((rho_s - rho) * g * (1-n))        # volumetric rate of longshore sediment transport [m^3/s]
Q_l_year = Q_l * 3600 * 24 * 365 * p                    # yearly volumetric rate [m^3/year]

# Print the results
print('\n--- TASK 5 ---')
print(f"Instantaneous rate of longshore sediment transport I ≈ {I_l:.3f} N/s")
print(f"Volumetric rate of longshore sediment transport Q_l ≈ {Q_l:.6f} m^3/s")
print(f"Yearly volumetric rate of longshore sediment transport Q_l_year ≈ {Q_l_year:.6f} m^3/year\n")