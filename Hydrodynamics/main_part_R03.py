'''
ASSIGNMENT 3
Storm Surge
'''

import numpy as np
import constants as const
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from waves import *
from main_part_R01 import Hs, eta_s, mask_in

beta = 1/200                    # beach slope
d = 0.2 * 10**-3                # grain size [m]
a_0 = 0                         # angle of the approaching wave [degrees]
g = 9.81                        # gravity [m/s^2]
x = 300 *10**3                  # distance from the shoreline [m]
rho = 1000                      # water density [kg/m^3]
rho_air = 1.2                 # air density [kg/m^3]

B = 10                          # Beaufort scale
u_10 = 0.836 * B**(3/2)         # wind speed at 10 m height [m/s]


# --------------------------------------------------------
# TASK 1 - Cross-shore depth average equation of momentum
# --------------------------------------------------------
eta_ = 0 





#plt.title(r'$\overline{\eta}$') 
#plt.show()

# --------------------------------------------------------
# TASK 2 - Wind-induced shear stress at the surface
# --------------------------------------------------------
# Estimate friction velocity
C_d = 0.001 * (1.1 + 0.035 * u_10)                                 # drag coefficient
u_star = np.sqrt(C_d) * u_10                                       # friction velocity [m/s]

# Estimate wind-induced shear stress at the surface
tau_w = rho_air * u_star**2                                           # wind-induced shear stress at the surface [N/m^2]

# Print the results
print('\n--- TASK 2 ---')
print(f"Wind-induced shear stress at the surface: {tau_w} N/m^2")


# --------------------------------------------------------
# TASK 3 - Momentum equation solution
# --------------------------------------------------------
# Domain and boundary condition
x_offshore    = 300000      # m (300 km)
x_shore       = 0.0
eta_offshore  = 0.0         # η(300 km) = 0


# Depth profiles
def D_A(x):
    return beta * x

def D_B(x):
    return (beta / 2.0) * x

def D_C(x):
    return np.where(x * beta < 40.0, beta * x, 40.0)

# ODE Solver
def solve_eta_correct(Dfunc, tau_w, rho, g, x_shore, x_offshore, eta_offshore):
    
    def ode(x, eta):
        # ODE: dη/dx = τw / (ρ g (D(x) + η))
        depth = Dfunc(x)+eta
        dη_dx = -tau_w / (rho * g * depth)          # we put minus as we change the coordinate system shore -> offshore
        return dη_dx
    
    #print(f"Integration: {x_offshore/1000:.0f}km -> {x_shore/1000:.0f}km")
    #print(f"tau_w/(ρg) = {tau_w/(rho*g):.2e}")
    
    # Integration with conservative parameters
    sol = solve_ivp(ode, 
                    (x_offshore, x_shore), 
                    [eta_offshore],
                    method='RK45',
                    dense_output=True,
                    rtol=1e-8,
                    atol=1e-10,
                    max_step=1000)  # maximum iteration change
    
    xs = np.linspace(x_offshore, x_shore, 1000)
    etas = sol.sol(xs)[0]
    etas= np.abs(etas)  

    return xs, etas

# SOLUTION
xA, etaA = solve_eta_correct(D_A, tau_w, rho, g, x_shore, x_offshore, eta_offshore)
xB, etaB = solve_eta_correct(D_B, tau_w, rho, g, x_shore, x_offshore, eta_offshore)
xC, etaC = solve_eta_correct(D_C, tau_w, rho, g, x_shore, x_offshore, eta_offshore)


# FINAL PLOT
plt.figure(figsize=(12, 8))

# Plot storm surge
plt.subplot(2, 2, 1)
plt.plot(xA/1000, etaA, 'b-', linewidth=2, label='A: slope β')
plt.plot(xB/1000, etaB, 'r-', linewidth=2, label='B: slope β/2')
plt.plot(xC/1000, etaC, 'g-', linewidth=2, label='C: β until 40m')
plt.xlabel('Distance from shore (km)')
plt.ylabel('Storm surge η (m)')
plt.title('Storm Surge - ODE Solution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()  # Shore to the left

# Depth plot
plt.subplot(2, 2, 2)
plt.plot(xA/1000, D_A(xA), 'b--', alpha=0.7, label='D(x) Case A')
plt.plot(xB/1000, D_B(xB), 'r--', alpha=0.7, label='D(x) Case B')
plt.plot(xC/1000, D_C(xC), 'g--', alpha=0.7, label='D(x) Case C')
plt.xlabel('Distance from shore (km)')
plt.ylabel('Depth D(x) (m)')
plt.title('Depth Profiles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()

# Plot theoretical dη/dx
plt.subplot(2, 2, 3)
x_test = np.linspace(x_shore, x_offshore, 100)
dη_dx_A = tau_w / (rho * g * D_A(x_test))
dη_dx_B = tau_w / (rho * g * D_B(x_test)) 
dη_dx_C = tau_w / (rho * g * D_C(x_test))

plt.plot(x_test/1000, dη_dx_A, 'b-', alpha=0.7, label='dη/dx Case A')
plt.plot(x_test/1000, dη_dx_B, 'r-', alpha=0.7, label='dη/dx Case B')
plt.plot(x_test/1000, dη_dx_C, 'g-', alpha=0.7, label='dη/dx Case C')
plt.xlabel('Distance from shore (km)')
plt.ylabel('Theoretical dη/dx')
plt.title('Theoretical Derivative (η << D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()

# Print the results
print('\n--- TASK 3 ---')
for i, (x, eta, case) in enumerate(zip([xA, xB, xC], [etaA, etaB, etaC], ['A', 'B', 'C'])):
    print(f"Case {case}:")
    print(f"  Offshore ({x[0]/1000:.0f}km): η = {eta[0]:.4f} m")
    print(f"  Shore ({x[-1]/1000:.0f}km): η = {eta[-1]:.4f} m")
    print(f"  Δη = {eta[-1] - eta[0]:.4f} m")

# --------------------------------------------------------
# TASK 4 - Total increase in eta 
# --------------------------------------------------------
# Top level to the shoreline (from R01)
eta_partial_1 = eta_s[mask_in][-1]

# Total eta
eta_total = eta_partial_1 + etaC[-1]

# Print the results
print('\n--- TASK 4 ---')
print(f"Total increase in mean water level η at the shoreline: {eta_total:.4f} m")


# --------------------------------------------------------
# TASK 5 - Protective dunes
# --------------------------------------------------------
freeboard = 0.6 # m

design_height = eta_total + freeboard

print('\n--- TASK 5 ---')
print(f"Recommended dune height at shoreline: {design_height:.2f} m")

