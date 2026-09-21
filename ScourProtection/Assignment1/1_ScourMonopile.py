import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import prettytable as pt


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
delta_g = 2.3       # Geometric standard deviation of sediment [-]
D = 9.0             # Diameter of the monopile [m]
h = 30.0            # Water depth [m]
ks = 2.5 * d50      # Roughness height [m]

# Sea_state conditions
V_calm = 0.5        # Current velocity in calm conditions [m/s]
Hs_calm = 1.5       # Significant wave height in calm conditions [m]
Tp_calm = 7.0       # Peak wave period in calm conditions [s]

V_norm = 0.8        # Current velocity in normal conditions [m/s]
Hs_norm = 3.5       # Significant wave height in normal conditions [m]
Tp_norm = 11.0      # Peak wave period in normal conditions [s]

V_storm = 1.1       # Current velocity in storm conditions [m/s]
Hs_storm = 8.8      # Significant wave height in storm conditions [m]
Tp_storm = 14.0     # Peak wave period in storm conditions [s]


# ======================================================================
# TASK 5: Estimate KC, 𝑈_cw , D/L and 𝜃_cw for all conditions
# ======================================================================

def get_details(V, Hs, Tp, h=h, D=D):

    # === Calculate the the KC numbers ===
    Tz = Tp / 1.3
    U_m_bed = (Hs / (2 * np.sqrt(2))) * np.sqrt(g/h) * np.exp(-((3.65/Tz) * np.sqrt(h/g))**2.1)

    KC_c = (V * Tp) / D
    KC_w = (U_m_bed * Tp) / D
    KC_tot = KC_c + KC_w

    # === Calculate the U_cw ===
    z = 0.5 * D                                # Mid-monopile diameter
    U_fc = V / (6 + (1/kappa) * np.log(h/ks))  # Friction velocity at the bed
    U_c = U_fc / kappa * np.log(30 * z / ks)   # Friction velocity at the monopile
    U_cw = U_c / (U_c + U_m_bed)

    # === Calculate the D/L ratio ===
    hw = h                          # Staying consistent with the scour handbook
    L0 = g * Tp**2 / (2 * np.pi)    # Initial guess for wavelength in deep water

    def dispersion_relation(L):
        return (g * Tp**2/(2*np.pi)) * np.tanh((2*np.pi*hw)/L) - L

    L = fsolve(dispersion_relation, L0)[0]

    DL_ratio = D / L

    # === Calculate the theta_cw === 
    U_m = U_m_bed               # For the consistency in the formulas
    a = U_m * Tp / (2 * np.pi)  # Orbital wave amplitude

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

    # 

    return KC_c, KC_w, KC_tot, U_cw, DL_ratio, theta_cw


# Calculate the details for calm, normal, and storm conditions
KC_c_calm, KC_w_calm, KC_tot_calm, U_cw_calm, DL_ratio_calm, theta_cw_calm = get_details(V_calm, Hs_calm, Tp_calm)
KC_c_norm, KC_w_norm, KC_tot_norm, U_cw_norm, DL_ratio_norm, theta_cw_norm = get_details(V_norm, Hs_norm, Tp_norm)
KC_c_storm, KC_w_storm, KC_tot_storm, U_cw_storm, DL_ratio_storm, theta_cw_storm = get_details(V_storm, Hs_storm, Tp_storm)

# Make a summary table with the results
summary_table = pt.PrettyTable()
summary_table.field_names = ["Condition", "KC_c", "KC_w", "KC_tot", "U_cw", "D/L", "theta_cw",]
summary_table.align["Condition"] = "l"
summary_table.add_rows([
    ["Calm", f"{KC_c_calm:.2f}", f"{KC_w_calm:.2f}", f"{KC_tot_calm:.2f}", f"{U_cw_calm:.2f}", f"{DL_ratio_calm:.5f}", f"{theta_cw_calm:.2f}"],
    ["Normal", f"{KC_c_norm:.2f}", f"{KC_w_norm:.2f}", f"{KC_tot_norm:.2f}", f"{U_cw_norm:.2f}", f"{DL_ratio_norm:.5f}", f"{theta_cw_norm:.2f}"],
    ["Storm", f"{KC_c_storm:.2f}", f"{KC_w_storm:.2f}", f"{KC_tot_storm:.2f}", f"{U_cw_storm:.2f}", f"{DL_ratio_storm:.5f}", f"{theta_cw_storm:.2f}"],

])

print("\nSummary of Scour Parameters")
print(summary_table)


# ==========================================================================
# TASK 6: Estimate equilibrium scour depth for all conditions and time-scale
# ==========================================================================

def estimate_scour_depth_and_time_scale(U_cw, KC, DL_ratio, theta_cw):
    S_max = 1.3 * D
    A = 0.03 + 8 * U_cw ** (1/(max(KC,0.5))+5)
    B = (6-5.8 * np.tanh(200*((DL_ratio)**1.9)))*np.exp(-4.7* U_cw)
    S_eq = S_max * (1 - np.exp(-A*(max(KC,0.5) - B)))

    # Scouring time scale
    psi = 8 * 10**-5 * KC **2.5
    omega = 1/375 * (h/D)**0.75

    # Set up conditions
    if KC < 4:
        KC = 4

    if U_cw < 0.44:
        ratio_T_star_s_theta = psi * ((0.18 / psi)**(1/0.44))**U_cw  
        T_star_s = psi * ((0.18 / psi)**(1/0.44))**U_cw * theta_cw**(-3/2)   # Non-dimensional timescale
    else: 
        ratio_T_star_s_theta = omega * U_cw ** (np.log(0.18 / omega) / np.log(0.44))
        T_star_s = omega * U_cw ** (np.log(0.18 / omega) / np.log(0.44)) * theta_cw**(-3/2)   # Non-dimensional timescale

    T_s = D**2 / (np.sqrt(g*(s-1)*d50**3)) * T_star_s

    # Backfilling time scale
    T_star_b = (1/50 + 0.015 * (np.exp(-350*(U_cw-0.5)**2)+np.exp(-25*(U_cw-0.53)**2))) * theta_cw**(-5/3)   # Non-dimensional timescale
    T_b = D**2 / (np.sqrt(g * (s - 1) * d50**3)) * T_star_b  

    return S_eq, ratio_T_star_s_theta, T_star_s, T_s, T_star_b, T_b, KC






S_eq_calm, ratio_T_star_s_theta_calm, T_star_calm, T_calm, T_star_b_calm, T_b_calm, KC_used_calm = estimate_scour_depth_and_time_scale(U_cw_calm, KC_tot_calm, DL_ratio_calm, theta_cw_calm)
S_eq_norm, ratio_T_star_s_theta_norm, T_star_norm, T_norm, T_star_b_norm, T_b_norm, KC_used_norm = estimate_scour_depth_and_time_scale(U_cw_norm, KC_tot_norm, DL_ratio_norm, theta_cw_norm)
S_eq_storm, ratio_T_star_s_theta_storm, T_star_storm, T_storm, T_star_b_storm, T_b_storm, KC_used_storm = estimate_scour_depth_and_time_scale(U_cw_storm, KC_tot_storm, DL_ratio_storm, theta_cw_storm)

scour_table = pt.PrettyTable()
scour_table.field_names = ["Condition", "Ratio T*_s/θ", "KC", "S_eq (m)", "T_star", "Time scale (s)"]
scour_table.align["Condition"] = "l"
scour_table.add_rows([
    ["Calm", f"{ratio_T_star_s_theta_calm:.2f}", f"{KC_used_calm:.2f}", f"{S_eq_calm:.2f}", f"{T_star_calm:.2e}", f"{T_calm:.2f}"],
    ["Normal", f"{ratio_T_star_s_theta_norm:.2f}", f"{KC_used_norm:.2f}", f"{S_eq_norm:.2f}", f"{T_star_norm:.2e}", f"{T_norm:.2f}"],
    ["Storm", f"{ratio_T_star_s_theta_storm:.2f}", f"{KC_used_storm:.2f}", f"{S_eq_storm:.2f}", f"{T_star_storm:.2e}", f"{T_storm:.2f}"],
])

print("\nEstimated Equilibrium Scour Depths and Time Scales")
print(scour_table)