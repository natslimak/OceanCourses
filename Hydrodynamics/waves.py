import numpy as np
import constants as const
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp


#-------SHOALING -------#

def cons_energy_flux(H, H_0, k, D):                                             #energy flux conservation
    return H_0**2 - H**2 * (1 + 2*k*D/np.sinh(2*k*D)) * np.tanh(k*D)

def dispersion(k, D, T):                                                         #dispersion relation
    omega = 2*np.pi/T
    return omega**2 - const.g * k * np.tanh(k*D)

def shoaling(D, H_0, k_0, T):                          #sholaing --> is the system that solve the two equations for array D 
    def system(x):
        H, k = x
        return np.array([cons_energy_flux(H, H_0, k, D),dispersion(k, D, T)], dtype=float)
    
    H, k = fsolve(system, x0=[H_0, k_0])
    return H, k


def breaking_point(Ds, H_0, k_0, T, KB=0.8, tol=1e-2):
   
    omega = 2*np.pi/T

    for D in Ds:
        Hg, kg = shoaling(D, H_0, k_0, T)
        ratio = Hg / D

        if abs(ratio - KB) < tol:
            Db, Hb, kb = D, Hg, kg
            cb = omega / kb
            return Db, Hb, kb, cb


    return None, None, None, None




#------- REFRACTION -------- #

def cons_energy_refraction (H, H_0, k, D, alpha, alpha_0): #energy flux conservation with refraction
    G = 2*k*D / np.sinh(2*k*D)
    K_s = np.sqrt(1.0 /(np.tanh(k*D) * (1.0+G)))
    K_r = np.sqrt(np.cos(alpha_0)/np.cos(alpha))
    return H - H_0*K_r*K_s


def Snell_law (alpha, alpha_0, k, k_0):                   #Snell's law --> k_0 * sen (alpha_0) = k * sen(alpha)
    return np.sin(alpha)*k - np.sin(alpha_0)*k_0



def refraction(D, H_0, k_0, alpha_0, T):
    def system(x):
        H, k, alpha = x
        return np.array([cons_energy_refraction(H, H_0, k, D, alpha, alpha_0),dispersion(k, D, T), Snell_law(alpha, alpha_0, k, k_0)], dtype=float)
    
    H, k, alpha = fsolve(system, x0=[H_0, k_0, alpha_0])
    return H, k, alpha


def breaking_point_refraction(Ds, H_0, k_0, alpha_0, T, KB=0.8, tol=1e-3):
    
    omega = 2*np.pi/T

    for D in Ds:
        Hg, kg, alphag = refraction(D, H_0, k_0, alpha_0, T)
        ratio = Hg / D

        if abs(ratio - KB) < tol:
            Db, Hb, kb, alphab = D, Hg, kg, alphag
            cb = omega / kb
            return Db, Hb, kb, cb, alphab
    
    return None, None, None, None, None


#------- MOMENTUM -------- #
def solve_eta(Dfunc, tau_w, rho, g, x_shore, x_offshore, eta_offshore, Dmin=0.5):   # minimum depth to avoid singularity
    # ODE: dη/dx = τw / (ρ g (D(x) + η))
    def ode(x, eta):
        depth = max(Dfunc(x) + eta, Dmin)
        return tau_w / (rho * g * depth)
    
    sol = solve_ivp(ode,(x_shore, x_offshore),[eta_offshore],dense_output=True,rtol=1e-7,atol=1e-9)
    
    xs = np.linspace(x_shore, x_offshore, 800)
    etas = sol.sol(xs)[0]
    return xs, etas
