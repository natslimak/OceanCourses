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
    """Calculate the KC numbers, U_cw, D/L ratio and theta_cw for a given set of conditions."""

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
    """Estimate the equilibrium scour depth and time-scale for a given set of conditions."""
    S_max = 1.3 * D
    A = 0.03 + 8 * U_cw ** (1/(max(KC,0.5))+5)
    B = (6-5.8 * np.tanh(200*((DL_ratio)**1.9)))*np.exp(-4.7* U_cw)
    S_eq = S_max * (1 - np.exp(-A*(max(KC,0.5) - B)))

    # Set up conditions
    if KC < 4:
        KC_temp = 4 # Temporary value as KC is less than 4 in the next formula
    else:
        KC_temp = KC

    # Scouring time scale
    psi = 8 * 10**-5 * KC_temp **2.5
    omega = 1/375 * (h/D)**0.75


    if U_cw < 0.44:
        ratio_T_star_s_theta = psi * ((0.18 / psi)**(1/0.44))**U_cw  
        T_star_s = psi * ((0.18 / psi)**(1/0.44))**U_cw * theta_cw**(-3/2)   # Non-dimensional timescale
    else: 
        ratio_T_star_s_theta = omega * U_cw ** (np.log(0.18 / omega) / np.log(0.44))
        T_star_s = omega * U_cw ** (np.log(0.18 / omega) / np.log(0.44)) * theta_cw**(-3/2)   # Non-dimensional timescale

    T_s = D**2 / (np.sqrt(g*(s-1)*d50**3)) * T_star_s

    # Backfilling time scale
    # FIXME: ask what happens if KC is between 2 and 3 (LEcture , slide 54)
    if KC < 2:
        upsilon = 0.31
    elif KC > 3:
        upsilon = 8.5 * KC ** (-0.5)
    else:
        print("Problem with upsilon")
        upsilon = 0

    lambda_big = 1 - 0.09*np.tanh(3000*U_cw**20-0.6**20)
    
    T_star_b = upsilon * lambda_big * theta_cw**(-3/2)
    T_b = D**2 / (np.sqrt(g * (s - 1) * d50**3)) * T_star_b  

    return S_eq,T_star_s, T_s, T_star_b, T_b, KC



# Print the results
S_eq_calm, T_star_calm, T_calm, T_star_b_calm, T_b_calm, KC_used_calm = estimate_scour_depth_and_time_scale(U_cw_calm, KC_w_calm, DL_ratio_calm, theta_cw_calm)
S_eq_norm, T_star_norm, T_norm, T_star_b_norm, T_b_norm, KC_used_norm = estimate_scour_depth_and_time_scale(U_cw_norm, KC_w_norm, DL_ratio_norm, theta_cw_norm)
S_eq_storm, T_star_storm, T_storm, T_star_b_storm, T_b_storm, KC_used_storm = estimate_scour_depth_and_time_scale(U_cw_storm, KC_w_storm, DL_ratio_storm, theta_cw_storm)

# Put them into a nice table 
scour_table = pt.PrettyTable()
scour_table.field_names = ["Condition","KC", "S_eq (m)", "T_star_s", "T_s[hrs]", "T_star_b", "T_b[hrs]"]
scour_table.align["Condition"] = "l"
scour_table.add_rows([
    ["Calm", f"{KC_used_calm:.2f}", f"{S_eq_calm:.2f}", f"{T_star_calm:.2e}", f"{T_calm/3600:.2f}", f"{T_star_b_calm:.2e}", f"{T_b_calm/3600:.2f}"],
    ["Normal", f"{KC_used_norm:.2f}", f"{S_eq_norm:.2f}", f"{T_star_norm:.2e}", f"{T_norm/3600:.2f}", f"{T_star_b_norm:.2e}", f"{T_b_norm/3600:.2f}"],
    ["Storm", f"{KC_used_storm:.2f}", f"{S_eq_storm:.2f}", f"{T_star_storm:.2e}", f"{T_storm/3600:.2f}", f"{T_star_b_storm:.2e}", f"{T_b_storm/3600:.2f}"],
])

print("\nEstimated Equilibrium Scour Depths and Time Scales")
print(scour_table)



# ==========================================================================
# TASK 7: Scour Development 
# ==========================================================================

def scour_development(t, S_eq, T_s):
    """Predict scour depth after installation while the pile is unprotected."""
    return S_eq * (1 - np.exp(-t / T_s))


conditions = {
    "Calm": (S_eq_calm, T_calm, "tab:blue"),
    "Normal": (S_eq_norm, T_norm, "tab:orange"),
    "Storm": (S_eq_storm, T_storm, "tab:red"),
}

# Making a common time axis long enough to show the development of every case
t_plot = np.linspace(0, 5 * max(T_calm, T_norm, T_storm), 400)

plt.figure(figsize=(8, 4.5), dpi=150)
for condition, (S_eq, T_s, colour) in conditions.items():
    plt.plot(
        t_plot / 3600,
        scour_development(t_plot, S_eq, T_s) / D,
        color=colour,
        linewidth=2,
        label=condition,
    )

plt.xlabel("Time after monopile installation [hours]")
plt.ylabel("$S/D$")
plt.title("Predicted scour development before scour protection")
plt.grid(True, alpha=0.3)
plt.legend(title="Flow condition")
plt.tight_layout()
plt.show()


# ==========================================================================
# TASK 8a: Depth averaged velocity over a 14 day period in Tidal Conditions
# ==========================================================================
V_sc  = 0.7     # Depth-averaged spring current [m/s]
V_nc = 0.5      # Depth-averaged neap current [m/s]
V_oc = 0.46     # Depth-averaged offset current [m/s]

T_ebb_flood = 12.5              # Ebb/flood current oscillation period [hours]
T_spring_neap = 14 * 24         # Spring/neap current oscillation period [hours]

# Current oscillation amplitudes from the spring, neap, and offset currents.
a_ebb_flood = (V_sc + V_nc) / 2 - V_oc
a_spring_neap = (V_sc - V_nc) / 2


def tidal_velocity(t):
    """Return depth-averaged tidal velocity at time t [hours]."""
    ebb_flood = a_ebb_flood * (2 * np.abs(np.cos(2 * np.pi * t / T_ebb_flood)) - 1)
    spring_neap = a_spring_neap * np.cos(2 * np.pi * t / T_spring_neap)
    return V_oc + ebb_flood + spring_neap


t_tidal = np.linspace(0, T_spring_neap, 2000) # make a time array 
velocity_tidal = tidal_velocity(t_tidal)

plt.figure(figsize=(8, 4.5), dpi=150)
plt.plot(t_tidal / 24, velocity_tidal, color="tab:blue", linewidth=1.2)
plt.xlabel("Time [days]")
plt.ylabel("Depth-averaged velocity, $V_c(t)$ [m/s]")
plt.title("Depth-averaged tidal velocity over 14 days")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTask 8: Tidal velocity parameters")
print(f"Ebb/flood amplitude: {a_ebb_flood:.3f} m/s")
print(f"Spring/neap amplitude: {a_spring_neap:.3f} m/s")


# ==========================================================================
# TASK 8b: Depth averaged velocity to predict scour development
# ==========================================================================

# For current-only flow, use the current-scour equilibrium depth and update
# the scour time scale with the instantaneous tidal velocity.
S_eq_current = 0.6 * D

def current_scour_time_scale(velocity):
    """Return the current-only scour time scale for velocity [m/s] in hours."""
    friction_velocity = velocity / (6 + (1 / kappa) * np.log(h / ks))
    theta_current = friction_velocity**2 / (g * d50 * (s - 1))
    T_star_current = (1 / 50) * theta_current**(-5 / 3)
    T_current = D**2 / np.sqrt(g * (s - 1) * d50**3) * T_star_current
    return T_current / 3600


def integrate_scour(t_end, dt=0.25):
    """Integrate dS/dt = (S_eq - S) / T_s(V_c(t)) from zero scour."""
    time = np.arange(0, t_end + dt, dt)
    scour = np.zeros_like(time)

    for index in range(1, len(time)):
        velocity = tidal_velocity(time[index - 1])
        time_scale = current_scour_time_scale(velocity)
        scour[index] = S_eq_current + (scour[index - 1] - S_eq_current) * np.exp(-dt / time_scale)

    return time, scour


t_week, scour_week = integrate_scour(7 * 24)
t_four_months, scour_four_months = integrate_scour(4 * 30 * 24)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharey=True)
axes[0].plot(t_week, scour_week / D, color="tab:green", linewidth=1.5)
axes[0].set_title("Current-only scour development over one week")
axes[0].set_xlabel("Time [hours]")
axes[0].set_ylabel("Scour depth, $S/D$")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_four_months / 24, scour_four_months / D, color="tab:purple", linewidth=1.5)
axes[1].set_title("Current-only scour development over four months")
axes[1].set_xlabel("Time [days]")
axes[1].set_ylabel("Scour depth, $S/D$")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTask 8b: Current-only scour development")
print(f"Equilibrium scour depth: {S_eq_current:.2f} m")
print(f"Scour after one week: {scour_week[-1]:.2f} m ({scour_week[-1] / D:.3f}D)")
print(
    f"Scour after four months: {scour_four_months[-1]:.2f} m "
    f"({scour_four_months[-1] / D:.3f}D)"
)
