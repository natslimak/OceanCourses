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




# ============================================================================
# TASK 9: Two-layer Scour Protection Design
# ============================================================================

t_filt = 0.9        # Filter layer thickness [m]
t_armour = 3*D      # Armour layer thickness [m]

# Stone specifications 
D50_stone = 90e-3       # Median stone size [m] CP63/180
htop = h - 1.9     # Water depth above the armour layer [m]
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
    print(f"KC_w: {KC_w:.3f}, KC_c: {KC_c:.3f}, KC_tot: {KC_tot:.3f}")


    # === Calculating bed shear stress and Shields parameter ===

    # Bed rounghness height
    ks = 2.5 * D50_stone

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

    print(f"u_star_w: {u_star_w:.3f}, f_w: {f_w:.3f}, A_wa/ks: {A_wa_ks:.3f}")

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
    alfa = 0
    tau_max = np.sqrt(tau_m ** 2 + tau_w ** 2 + 2 * tau_m * tau_w * (np.cos((alfa * np.pi)/180)))

    print(f"tau_max: {tau_max:.3f}, tau_m: {tau_m:.3f}, tau_w: {tau_w:.3f}, tau_c: {tau_c:.3f}")

    # === Calculating the mobility number ===

    # specific rock density
    delta_s = rho_stone - rho / rho

    # dimensionless particle diameter
    d_star = d50 * (delta_s * g / nu**2)**(1/3)

    # Critical Shields parameter
    theta_cr = 0.3 / (1 + 1.2*d_star)+0.055*(1-np.exp(-0.02*d_star))

    # Combined waves and current Shields parameter
    theta_cw = tau_max / ((rho_s - rho) * g * D50_stone)
    print(f"theta_cw: {theta_cw:.3f}, theta_cr: {theta_cr:.3f}")

    # Mobility number
    MOB = theta_cw / theta_cr



    # === Estimating depth of deformation ===
    f_KCtot = 1 + (3.9274 / (1 + np.exp(-0.7401 * KC_tot + 4.7518)))
    
    # Depth of deformation
    S_90perc = D * f_KCtot * (0.1134 * MOB ** 1.6492)
    print(f"MOB: {MOB:.3f}, f_KCtot: {f_KCtot:.3f}\nS_90perc: {S_90perc:.3f} m\n")
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