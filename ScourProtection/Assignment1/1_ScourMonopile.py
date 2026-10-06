import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
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
    KC = (U_m_bed * Tp) / D

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

    return KC, U_cw, DL_ratio, theta_cw


# Calculate the details for calm, normal, and storm conditions
KC_calm,  U_cw_calm, DL_ratio_calm, theta_cw_calm = get_details(V_calm, Hs_calm, Tp_calm)
KC_norm, U_cw_norm, DL_ratio_norm, theta_cw_norm = get_details(V_norm, Hs_norm, Tp_norm)
KC_storm, U_cw_storm, DL_ratio_storm, theta_cw_storm = get_details(V_storm, Hs_storm, Tp_storm)

# Make a summary table with the results
summary_table = pt.PrettyTable()
summary_table.field_names = ["Condition", "KC", "U_cw", "D/L", "theta_cw",]
summary_table.align["Condition"] = "l"
summary_table.add_rows([
    ["Calm", f"{KC_calm:.2f}", f"{U_cw_calm:.2f}", f"{DL_ratio_calm:.5f}", f"{theta_cw_calm:.2f}"],
    ["Normal", f"{KC_norm:.2f}", f"{U_cw_norm:.2f}", f"{DL_ratio_norm:.5f}", f"{theta_cw_norm:.2f}"],
    ["Storm", f"{KC_storm:.2f}", f"{U_cw_storm:.2f}", f"{DL_ratio_storm:.5f}", f"{theta_cw_storm:.2f}"],

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
        T_star_s = psi * ((0.18 / psi)**(1/0.44))**U_cw * theta_cw**(-3/2)   # Non-dimensional timescale
    else: 
        T_star_s = omega * U_cw ** (np.log(0.18 / omega) / np.log(0.44)) * theta_cw**(-3/2)   # Non-dimensional timescale

    T_s = D**2 / (np.sqrt(g*(s-1)*d50**3)) * T_star_s

    # Backfilling time scale
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

    return S_eq, T_star_s, T_s, T_star_b, T_b, KC



# Print the results
S_eq_calm, T_star_calm, T_calm, T_star_b_calm, T_b_calm, KC_used_calm = estimate_scour_depth_and_time_scale(U_cw_calm, KC_calm, DL_ratio_calm, theta_cw_calm)
S_eq_norm, T_star_norm, T_norm, T_star_b_norm, T_b_norm, KC_used_norm = estimate_scour_depth_and_time_scale(U_cw_norm, KC_norm, DL_ratio_norm, theta_cw_norm)
S_eq_storm, T_star_storm, T_storm, T_star_b_storm, T_b_storm, KC_used_storm = estimate_scour_depth_and_time_scale(U_cw_storm, KC_storm, DL_ratio_storm, theta_cw_storm)

# Put them into a nice table 
scour_table = pt.PrettyTable()
scour_table.field_names = ["Condition","KC", "S_eq (m)", "T_star_s", "T_s[days]", "T_star_b", "T_b[days]"]
scour_table.align["Condition"] = "l"
scour_table.add_rows([
    ["Calm", f"{KC_used_calm:.2f}", f"{S_eq_calm:.2f}", f"{T_star_calm:.2e}", f"{T_calm/(24 * 3600):.1f}", f"{T_star_b_calm:.2e}", f"{T_b_calm/(24 * 3600):.1f}"],
    ["Normal", f"{KC_used_norm:.2f}", f"{S_eq_norm:.2f}", f"{T_star_norm:.2e}", f"{T_norm/(24 * 3600):.1f}", f"{T_star_b_norm:.2e}", f"{T_b_norm/(24 * 3600):.1f}"],
    ["Storm", f"{KC_used_storm:.2f}", f"{S_eq_storm:.2f}", f"{T_star_storm:.2e}", f"{T_storm/(24 * 3600):.1f}", f"{T_star_b_storm:.2e}", f"{T_b_storm/(24 * 3600):.1f}"],
])

print("\nEstimated Equilibrium Scour Depths and Time Scales")
print(scour_table)



# ==========================================================================
# TASK 7: Scour Development 
# ==========================================================================

# Scour development function S(t)
scour_dev = lambda t, S_eq, T_s: S_eq * (1 - np.exp(-t / T_s))

# Prepare for the plotting
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
        t_plot / (24 * 3600),
        scour_dev(t_plot, S_eq, T_s) / D,
        color=colour,
        linewidth=2,
        label=condition,
    )

plt.xlabel("Time after monopile installation [days]")
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

T_ebb_flood = 12.5 * 3600       # Ebb/flood current oscillation period [s]
T_spring_neap = 14 * 24 * 3600  # Spring/neap current oscillation period [s]

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
plt.plot(t_tidal / (24 * 3600), velocity_tidal, color="tab:blue", linewidth=1.2)
plt.xlabel("Time [days]")
plt.ylabel("Depth-averaged velocity, $V_c(t)$ [m/s]")
#plt.title("Depth-averaged tidal velocity over 14 days")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# ==========================================================================
# TASK 8b: Depth averaged velocity to predict scour development + no waves
# ==========================================================================
S_eq_c = 1.3 * D

def current_scour_time_scale(V):
    """Return the current-only scour time scale for velocity [m/s] in seconds."""
    U_fc = V / (6 + (1 / kappa) * np.log(h / ks))
    theta_c = U_fc**2 / (g * d50 * (s - 1))
    T_star_c = (1 / 375) * (h/D) ** 0.75 * theta_c ** (-3 / 2)
    T_c = D**2 / np.sqrt(g * (s - 1) * d50**3) * T_star_c
    return T_c


def integrate_scour(t_end, dt=500):
    """Integrate dS/dt = (S_eq - S) / T_s(V_c(t)) from zero scour."""
    def scour_ode(time, scour):
        V_s = tidal_velocity(time)
        T_s = current_scour_time_scale(V_s)

        return (S_eq_c - scour[0]) / T_s

    time = np.arange(0, t_end + dt / 2, dt)
    solution = solve_ivp(
        scour_ode,
        (time[0], time[-1]),
        y0=[0.0],
        t_eval=time,
        max_step=dt,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.t, solution.y[0]



t_week_c, scour_week_c = integrate_scour(7 * 24 * 3600)
t_four_months_c, scour_four_months_c = integrate_scour(4 * 30 * 24 *3600)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharey=True)
axes[0].plot(t_week_c / (24 * 3600), scour_week_c, color="tab:green", linewidth=1.5)
axes[0].set_title("Time Varying Current-only scour development over one week")
axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel("Scour depth, $S$")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_four_months_c / (24 * 3600), scour_four_months_c, color="tab:purple", linewidth=1.5)
axes[1].set_title("Time Varying Current-only scour development over four months")
axes[1].set_xlabel("Time [days]")
axes[1].set_ylabel("Scour depth, $S$")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTask 8b: Current-only scour development")
print(f"Equilibrium scour depth: {S_eq_c:.2f} m")
print(f"Scour after one week: {scour_week_c[-1]:.2f} m ({scour_week_c[-1] / D:.3f}D)")
print(
    f"Scour after four months: {scour_four_months_c[-1]:.2f} m "
    f"({scour_four_months_c[-1] / D:.3f}D)"
)


# ============================================================================
# TASK 8c: Depth averaged velocity to predict scour development + waves (calm)
# ============================================================================

def current_and_waves_scour_time_scale(V, scour):
    """Return the current-only scour time scale for velocity [m/s] in seconds."""

    # Get the current and wave parameters for the varying current velocity
    KC_now, U_cw_now, DL_ratio_now, theta_cw_now = get_details(V, Hs_calm, Tp_calm)

    # Calculate time-scale for scour and backfilling 
    S_eq_varied, T_star_s, T_s, T_star_b, T_b, KC = estimate_scour_depth_and_time_scale(U_cw_now, KC_now, DL_ratio_now, theta_cw_now)

    # Evaluate whether the scour or backfilling time-scale is larger and return the larger one
    if S_eq_varied > scour:
        return T_s, S_eq_varied
    elif S_eq_varied <= scour:
        return T_b, S_eq_varied


def integrate_scour_cw(t_end, dt=500):
    """Integrate dS/dt = (S_eq - S) / T_s(V_c(t)) from zero scour."""
    def scour_ode(time, scour):
        V_s = tidal_velocity(time)
        T, S_eq_varied = current_and_waves_scour_time_scale(V_s, scour[0])
        return (S_eq_varied - scour[0]) / T

    time = np.arange(0, t_end + dt / 2, dt)
    solution = solve_ivp(
        scour_ode,
        (time[0], time[-1]),
        y0=[0.0],
        t_eval=time,
        max_step=dt,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.t, solution.y[0]



t_week_cw, scour_week_cw = integrate_scour_cw(7 * 24 * 3600)
t_four_months_cw, scour_four_months_cw = integrate_scour_cw(4 * 30 * 24 *3600)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharey=True)
axes[0].plot(t_week_cw / (24 * 3600), scour_week_cw, color="tab:green", linewidth=1.5)
axes[0].set_title("Time Varying Current + Calm Waves scour development over one week")
axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel("Scour depth, $S$")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_four_months_cw / (24 * 3600), scour_four_months_cw, color="tab:purple", linewidth=1.5)
axes[1].set_title("Time Varying Current + Calm Waves scour development over four months")
axes[1].set_xlabel("Time [days]")
axes[1].set_ylabel("Scour depth, $S$")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTask 8c: Current+waves scour development")
print(f"Scour after one week: {scour_week_cw[-1]:.2f} m ({scour_week_cw[-1] / D:.3f}D)")
print(
    f"Scour after four months: {scour_four_months_cw[-1]:.2f} m "
    f"({scour_four_months_cw[-1] / D:.3f}D)"
)

# === Summary of all three scour developments ===

plt.figure(figsize=(8, 4.5), dpi=150)
#plt.title("Comparison of scour development under different conditions")
# Use one common time axis and interpolate the tidal results onto it.
t_max_plot = max(t_plot[-1], t_four_months_c[-1], t_four_months_cw[-1])
t_summary = np.linspace(0, t_max_plot, 400)

# Calm condition scour development
plt.plot(
    t_summary / (24 * 3600),#
    scour_dev(t_summary, S_eq_calm, T_calm) / D,
    color="tab:blue",
    linewidth=2,
    label="Calm Current + Calm Waves",
)
# Current-only scour development
plt.plot(
    t_four_months_c / (24 * 3600),
    scour_four_months_c / D,
    color="tab:green",
    linewidth=2,
    label="Time-varying Current",
)
# Current+waves scour development
plt.plot(
    t_four_months_cw / (24 * 3600),
    scour_four_months_cw / D,
    color="tab:purple",
    linewidth=2,
    label="Time-varying Current + Calm Waves",
)
plt.xlabel("Time [days]")
plt.ylabel("Scour depth, $S/D$")
plt.legend(title="Scour development conditions:")


# ============================================================================
# TASK 9: Two-layer Scour Protection Design
# ============================================================================

t_filt = 0.9        # Filter layer thickness [m]
t_armour = 3*D      # Armour layer thickness [m]

# Stone specifications 
D50_stone = 90e-3       # Median stone size [m] CP63/180
htop = 1.0              # Water depth above the armour layer [m]
rho_stone = 3200        # Stone density [kg/m^3]


def get_details_HASPRO(V, Hs, Tp, h=h, D=D, htop=htop):
    """Calculate the KC numbers, U_cw, D/L ratio and theta_cw for a given set of conditions."""

    # Statying consistent with the scour handbook
    hw = h


    # === Calculating hydrodynamic parameters ===

    # Wave velocity near the bed
    Tz = Tp / 1.3
    U_m_bed = (Hs / (2 * np.sqrt(2))) * np.sqrt(g/h) * np.exp(-((3.65/Tz) * np.sqrt(h/g))**2.1)
    
    # Wave lenght
    L0 = g * Tp**2 / (2 * np.pi)    # Initial guess for wavelength in deep water
    def dispersion_relation(L):
        return (g * Tp**2/(2*np.pi)) * np.tanh((2*np.pi*hw)/L) - L
    L = fsolve(dispersion_relation, L0)[0]
        
    # Corerction factor
    K_top = (np.sinh(2 * np.pi * hw / L) ** 2) / (np.sinh(2 * np.pi * htop / L))

    # Wave velocity on top of the scour protection
    U_m_top = K_top * U_m_bed

    # Wave orbital motion amplitude on top of the scour protection
    A_wtop = U_m_top * Tp / (2 * np.pi)
    A_wa = A_wtop   

    # Get the KC numbers
    KC_w = (U_m_bed * Tp) / D
    KC_c = (V * Tp) / D # FIXME Is the U_c here V?
    KC_tot = KC_w + KC_c



    # === Calculating bed shear stress and Shields parameter ===

    # Ratio of wave orbital motion amplitude to roughness height
    A_wa_ks = A_wa / ks

    # Wave friction factor (Roulund's roughness)
    if 0.2 < A_wa / ks < 2.92:
        f_w = 0.32 * (A_wa / ks) ** (-0.8)
    elif 2.92 <= A_wa / ks < 727:
        f_w = 0.237 * (A_wa / ks) ** (-0.52)
    elif A_wa / ks >= 727:
        f_w = 0.04 * (A_wa / ks) ** (-0.25)

    # Wave-related shear velocity 
    u_star_w = np.sqrt(f_w / 2) * U_m_top

    # wave bed shear stress 
    tau_w = rho * u_star_w ** 2

    # Current induced bed shear stress
    z_0 = ks / 30

    # Drag coefficient
    C_D = (0.4 / (np.log(hw / z_0)-1)) ** 2

    # Current shear velocity
    u_star_c = np.sqrt(C_D) * V

    # Current shear stress
    tau_c = rho * u_star_c ** 2

    # The mean combined current and wave bed shear stress
    tau_m = tau_c * (1.2 * tau_c * (tau_w / (tau_c + tau_w)) ** (3.2))

    # The maximum combined current and wave bed shear stress
    alfa = 0
    tau_max = tau_m + tau_w 



    # === Calculating the mobility number ===

    # specific rock density
    delta_s = rho_stone - rho / rho

    # dimensionless particle diameter
    d_star = d50 * (delta_s * g / nu**2)**(1/3)

    # Critical Shields parameter
    theta_cr = 0.3 / (1 + 1.2*d_star)+0.055*(1-np.exp(-0.02*d_star))

    # Combined waves and current Shields parameter
    theta_cw = tau_max / ((rho_s - rho) * g * D50_stone)

    # Mobility number
    MOB = theta_cw / theta_cr



    # === Estimating depth of deformation ===
    f_KCtot = 1 + (3.9274 / (1 + np.exp(-0.7401 * KC_tot + 4.7518)))

    # Depth of deformation
    S_90perc = D * f_KCtot * (MOB ** 1.6492)

    return MOB, S_90perc


# Get the details for calm, normal, and storm conditions
MOB_calm, S_90perc_calm = get_details_HASPRO(V_calm, Hs_calm, Tp_calm)
MOB_normal, S_90perc_normal = get_details_HASPRO(V_norm, Hs_norm, Tp_norm)
MOB_storm, S_90perc_storm = get_details_HASPRO(V_storm, Hs_storm, Tp_storm)

# Make a summary table with the results
HASPRO_table = pt.PrettyTable()
HASPRO_table.field_names = ["Condition", "Mobility Number", "Depth of Deformation"]
HASPRO_table.align["Condition"] = "l"
HASPRO_table.add_rows([
    ["Calm", f"{MOB_calm:.2f}", f"{S_90perc_calm:.2f} m"],
    ["Normal", f"{MOB_normal:.2f}", f"{S_90perc_normal:.2f} m"],
    ["Storm", f"{MOB_storm:.2f}", f"{S_90perc_storm:.2f} m"],
])

print("\nTwo-layer Scour Protection Design Results")
print(HASPRO_table)