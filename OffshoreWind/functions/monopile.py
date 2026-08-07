import numpy as np
from common import loadConstants

g = loadConstants()["g"]
rho_water = loadConstants()["rho_water"]

def forceIntegrate(monopileDict, u, ut, z, x_dot, z_ref=None):
    
    h = np.abs(z[0])
    u = u - x_dot
    
    df = forceDistributed(monopileDict, u, ut, z, x_dot)
    
    F = np.trapezoid(df, z)
    # Total moment around specified reference point
    if z_ref is None:
        z_ref = 0  # Default: moment at seabed (mudline)
    
    M = np.trapezoid(df * (z - z_ref), z)   # df * moment arm from z_ref

    return F, M

def forceDistributed(monopileDict, u, ut, z, x_dot):
    
    u = u - x_dot   # u -> horizontal velocity of the water particles, x_dot -> horizontal velocity of the sturcture itself
    
    # FIXME Assignment 1 Q2.3:  add back the inertia forces
    A = (monopileDict["DMonopile"]/2)**2*np.pi
    df = 0.5*rho_water*monopileDict["DMonopile"]*monopileDict["CD"]*np.abs(u)*u + rho_water * monopileDict["CM"] *A*ut

    return df

def computeElementwiseQuantities(monopileDict):
    
    outputDict = dict()
    outputDict.update(monopileDict)
    
    # Compute missing element properties
    z = monopileDict["zBeamNodal"]
    dz = np.diff(z)
    outputDict["zBeamElement"] = z[:-1] + dz/2
    outputDict["dz"] = dz
    
    # Compute the phiNodal
    phi = monopileDict["phiNodal"]
    dPhi = np.diff(phi)
    outputDict["phiElement"] = outputDict["phiNodal"][:-1] + dPhi/2
    
    return outputDict
    
