import matplotlib.pyplot as plt
import numpy as np


# ======================================================================
# Given data
# ======================================================================
# Constants
rho = 1026          # Water density [kg/m^3]
rho_s = 2650        # Sediment density [kg/m^3]
g = 9.81            # Gravitational acceleration [m/s^2]
kappa = 0.4         # von Karman constant [-]
nu = 1e-6           # Kinematic viscosity of water [m^2/s]
s = rho_s / rho     # Relative density of sediment [-]

# Given parameters
n = 0.43            # Porosity [-]
d50 = 0.2e-3        # Median grain size [m]
D = 0.8             # Diameter of the pipeline [m]
e = 0.04            # Initial Burial Depth [m]
h = 15.0            # Water depth [m]
ks = 2.5 * d50      # Roughness height [m]

# Spring tide current velocity properties
V = 1.1             # Depth-averaged current velocity [m/s]

# Dominant wave properties
Tp = 9.0            # Peak wave period [s]
Hs = 4.0            # Significant wave height [m]



# ======================================================================
# TASK 1a: Onset Scour - Steady Current
# ======================================================================

# Top of the pipeline (from the bottom)
z = D - e

# Define a range of burial ratios to plot the onset criteria as curves
ratios = np.geomspace(1e-3, 1, 200)

# Get the friction velocity in current flow
U_fc = V / (6 + (1/kappa) * np.log(h/ks))

# Get the velocity at the given depth
U = U_fc / kappa * np.log(30 * z / ks)

# Calculate the onset criteria
L_side_c = U **2 / g * D * (s - 1) * (1 - n)
R_side_c = lambda ratio: 0.025 * np.exp(9 * np.sqrt(ratio)) # ratio = e / D


# Plot the data
plt.figure(figsize=(8, 4), dpi=150)
plt.plot(ratios, R_side_c(ratios), color='red', linewidth=2)
plt.plot(e / D, L_side_c, 'o', label='Case 1', color='blue', markersize=7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.2g}'))
plt.xlabel(r'$e/D$', fontsize=12)
plt.ylabel(r'$\frac{U^{2}}{gD(s-1)(1-n)}$', rotation=90, labelpad=16, fontsize=12)
plt.title('Onset Scour: Steady Current', fontsize=14)
plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# Get the results
print('Results for Onset Scour - Steady Current:')
print(f'L_side_c: {L_side_c:.4f}', '   ', f'R_side_c: {R_side_c(e / D):.4f}')
print('\nOnset scour occurs if L_side_c > R_side_c:')
print(f'Value of L_side / R_side: {L_side_c / R_side_c(e / D):.4f}')
print('Onset scour status: ', 'Yes' if L_side_c > R_side_c(e / D) else 'No')


# === Test of the other function === 
f = 0.025 * np.exp(9 * np.sqrt(e / D))
dp_dx = - rho * U ** 2 / (D * f)
onset_scour = np.abs(dp_dx) / (g * rho * (s-1) * (1-n))

print('\nOnset scour occursif pressure gradient / (g * rho * (s-1) * (1-n)) > 1:')
print(f'Onset scour value: {onset_scour:.2f}') 
print('Onset scour status: ', 'Yes' if onset_scour > 1 else 'No')


# ======================================================================
# TASK 1b: Onset Scour - Waves
# ======================================================================

# Zero-crossing wave period
Tz = Tp / 1.3

# Free stream velocity at the top of the pipeline
U_m = (Hs / (2 * np.sqrt(2))) * np.sqrt(g/h) * np.exp(-((3.65/Tz) * np.sqrt(h/g))**2.1)

# Calculate the onset criteria
L_side_w = U_m **2 / (g * D * (s - 1) * (1 - n))
R_side_w = lambda ratio: 0.025 * np.exp(9 * np.sqrt(ratio)) # ratio = e / D

# Get the KC number
KC = U_m * Tp / D


# ======================================================================
# TASK 2: Primary Migration Speed of the Scour Hole
# ======================================================================
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

# Shields parameter 
theta_w = U_fw ** 2 / (g * d50 * (s - 1))          # For the waves
theta_c = U_fc ** 2 / (g * d50 * (s - 1))          # For the current

# Increased mean bed shear stress
theta_m = theta_c * (1 + 1.2 *(theta_w / (theta_c+theta_w)**(3/2)))

# Maximum combined shields parameter
theta_cw = theta_m + theta_w

# Friction velocity for the combined waves and current
U_fcw = np.sqrt(theta_cw * g * d50 * (s - 1))

# Reynolds number for the combined waves and current
Re_cw = U_fcw * d50 / nu

# Critical Shields parameter for initiation of motion
theta_cr_cw = 0.165 * (Re_cw + 0.6) ** (-0.8) + 0.045 * np.exp(-40 * (Re_cw ** (-1.3)))

# Velocities in the middle of the pipe
z_mid = D / 2 - e 
U_c_mid = U_fc / kappa * np.log(30 * z_mid / ks)
U_m_mid = (Hs / (2 * np.sqrt(2))) * np.sqrt(g/h) * np.exp(-((3.65/Tz) * np.sqrt(h/g))**2.1)

# m parameter 
m = U_c_mid / (U_m_mid + U_c_mid)

# Gamma parameter
big_gamma = 1 +200 * (theta_cw - theta_cr_cw) ** 3/2

# Gamma parameter
if m < 0.2: 
    small_gamma = 0.3
else: 
    small_gamma = 5.3 * np.exp(-2.96 * (m - 1.75) ** 2) + 6.1 * np.exp(-5.25 * (m + 0.56) ** 2)

# Big Lambda
big_lambda = np.exp(-3.2 * (e / D))

# Dimensionless primary migration speed of the scour hole
V_h_star = 3 * big_gamma * big_lambda * small_gamma

# Actual primary migration speed of the scour hole
V_h = (V_h_star * np.sqrt(g * (s - 1) * d50)) * (d50 / D)

# Time for the scour hole to reach 100D (divided by 2 because the scour hole migrates in both directions)
t_100 = 100 * D / (2 * V_h)

# Print the results
print('\nResults for Primary Migration Speed of the Scour Hole:')
print(f'Velocity of the scour hole: {V_h:.4f} m/s')
print(f'Time for the scour hole to reach 100D: {t_100/3600:.2f} hours')



# ======================================================================
# TASK 3: 2D Equilibrium Scour Profile
# ======================================================================

# === CASE 1: Waves ===
S_eq_w = 0.1 * np.sqrt(KC) * D
W_w = 0.35 * KC**0.65 * D

print('\n2D Equilibrium Scour Profile - Waves:')
print(f'Depth: {S_eq_w:.2f} m')
print(f'Width 1: {W_w:.2f} m')
print(f'Width 2: {W_w:.2f} m')


# === CASE 2: Steady Current ===
S_eq_c = 0.6 * D
W_1_c = 2 * D
W_2_c = 4 * D

print('\n2D Equilibrium Scour Profile - Current:')
print(f'Depth: {S_eq_c:.2f} m')
print(f'Width 1: {W_1_c:.2f} m')
print(f'Width 2: {W_2_c:.2f} m')


# === CASE 3: Tidal Current ===
S_eq_t = 0.6 * D
W_1_t = 4 * D
W_2_t = 4 * D

print('\n2D Equilibrium Scour Profile - Tidal Current:')
print(f'Depth: {S_eq_t:.2f} m')
print(f'Width 1: {W_1_t:.2f} m')
print(f'Width 2: {W_2_t:.2f} m')


# === CASE 4: Current + Waves ===

# Calculate the a 
if 0 <= m <= 0.4:
    a_m = 0.557 - 0.912 * (m - 0.25) ** 2
    b_m = -1.14 + 2.24 * (m - 0.25) ** 2
elif 0.4 < m <= 0.7:
    a_m = -2.14 * m + 1.46
    b_m = 3.3 * m - 2.5

# Calculate F 
if 0 <= m <= 0.7:
    F = (5/3) * (KC** a_m) * np.exp(2.3 * b_m)
elif m > 0.7:
    F = 1.0
else:
    raise ValueError("Invalid value of m")

S_eq_cw = 0.6 * D * F
W_1_cw = 1.9 * D
W_2_cw = 3.8 * D


print('\n2D Equilibrium Scour Profile - Current + Waves:')
print(f'Depth: {S_eq_cw:.2f} m')
print(f'Width 1: {W_1_cw:.2f} m')
print(f'Width 2: {W_2_cw:.2f} m')



# ======================================================================
# TASK 4: Scour Depth Development 
# ======================================================================
print('\nScour Depth Development:')
print(f'Scour depth - Current: {S_eq_c:.2f} m')
print(f'Scour depth - Current + Waves: {S_eq_cw:.2f} m')

# Climate parameters
t_c = 14 * 3600           # Time of current action [s]
t_w = 14 * 3600           # Time of wave action [s]


# SCOUR IN CURRENT CONDITION
T_star_c = 1/50 * theta_c**(-5/3)                           # Non-dimensional timescale 
T_c = D**2 / (np.sqrt(g * (s - 1) * d50**3)) * T_star_c     # Dimensional timescale for scour development [s]
S_tc = lambda t: S_eq_c * (1 - np.exp(-t/T_c))              # Dimensional scour depth development [m]
S_tc_at_tc = S_tc(t_c)                                      # Scour depth at time t_c [m]   
print(f'\nEvolution of scour after current action: {S_tc_at_tc:.2f} m')


# SCOUR IN WAVE CONDITION WITH BACKFILLING -> S_eq_cw < S_tc_at_tc
T_star_b = (1/50 + 0.015 * (np.exp(-350*(m-0.5)**2)+np.exp(-25*(m-0.53)**2))) * theta_cw**(-5/3)   # Non-dimensional timescale
T_b = D**2 / (np.sqrt(g * (s - 1) * d50**3)) * T_star_b                                            # Dimensional timescale for scour development [s]
S_tw = lambda t: S_eq_cw + (S_tc_at_tc - S_eq_cw) * np.exp(-t/T_b)                                 # Dimensional scour depth development [m]
S_t_cw = S_eq_cw * (1 - np.exp(-t_w/T_b))                                                          # Dimensional scour depth development [m] after wave action
print(f'\nEvolution of scour after wave action: {S_t_cw:.2f} m')


# Initialize the time arrays for plotting
current_time = np.linspace(0, t_c, 100)
wave_time = np.linspace(0, t_w, 100)

# Make the plot of the scour depth development
plt.figure(figsize=(8, 4), dpi=150)
plt.plot(current_time, S_tc(current_time), color='blue', linewidth=2, label=r'$S_c(t)$ - Steady Current')
plt.plot(t_c + wave_time, S_tw(wave_time), color='red', linewidth=2, label=r'$S_w(t)$ - Wave Dominant')
plt.xlabel('Time [s]', fontsize=12, labelpad=16) 
plt.ylabel('Scour Depth [m]', rotation=90, labelpad=16, fontsize=12)
#plt.title('Timescale of the Scour Depth Development', fontsize=14)
plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.legend()

