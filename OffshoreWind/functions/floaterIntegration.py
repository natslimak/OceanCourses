from integration import lookup
import numpy as np
from scipy.integrate import trapezoid
from loads import forceDistributed
from rotor import *
from bisect import bisect_left as lookup
from floatingRotor import F_wind as F_wind_floating

def dqdt(t, q,
                structure,
                rotor,
                waves,
                wind):
    
    x1 = q[0:2]
    xdot1 = q[2:4]
    CT1 = q[4]
    
    rotor["CT"] = CT1

    # Extract time index
    i_ = lookup(waves["t"], t)
    
    # Read wind speed
    V_hub = wind["V_hub"][i_]
    V_10 = wind["V_10"]
    
    # Nacelle speed
    x_dot_rotor = xdot1[0] + structure["zhub"]*xdot1[1]

    # Wind force if the rotor is on
    Thrust, CTVrel = F_wind_floating(rotor, V_10, V_hub, x_dot_rotor)
    Faero = np.array([Thrust, Thrust*structure["zhub"]])
    
    # FIXME Assignment 5 Q1.9: for xdot1
    #x_dot_submerged = 0. + structure["z"]*0.
    x_dot_submerged = xdot1[0] + structure["z"] * xdot1[1]  # x_0(dot) + z * theta(dot) <-- velocity
    
    u, ut = waves["u"][i_,:], waves["ut"][i_,:]
    df = forceDistributed(structure, u, ut, structure["z"], x_dot_submerged)
    Fhydro = np.array([trapezoid(df, structure["z"]),
                       trapezoid(df*structure["z"], structure["z"])])
    
    output = np.zeros(5)
    output[0:2] = xdot1
    # FIXME Assignment 5 Q1.9: for Equation of motion below
    #output[2:4] = np.random.rand(2)
    #(M+A)^-1  * (Faero + Fhydro - B*xdot1 - C*x1)
    #output[2:4] = (np.linalg.inv(structure["M"] + structure["A"])) @ (Faero + Fhydro - structure["B"] @ xdot1 - structure["C"] @ x1 )
    output[2:4] = np.linalg.solve(structure["M"] + structure["A"], Faero + Fhydro - structure["B"].dot(xdot1) - structure["C"].dot(x1))
    # FIXME Assignment 5 Q3.17: for control equation below
    F, CT_Vrel = F_wind_floating(rotor, V_10, V_hub, x_dot_rotor)
    output[4] = -rotor['gamma']*(q[4]-CT_Vrel)
    
    return output
