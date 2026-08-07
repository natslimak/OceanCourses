'''
Filename: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\src\\assignmentcode\\edition2024\\python\\loads.py
Path: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\src\\assignmentcode\\edition2024\\python
Created Date: Monday, October 7th 2024, 11:59:34 am
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''
import numpy as np
from rotor import *
from monopile import forceDistributed


def calculateStaticWindLoads(wind, rotor, structure, q):
       
    # Calculate the structural velocity at hub height
    phiHub = structure["phiNodal"][-1]
    x_dot_hub = phiHub*q["alphaDot"]
    
    F_with_rel_motion = np.zeros_like(wind["t"])
    
    for i_, t_ in enumerate(wind["t"]):
        # FIXME Assignment 3 Q1.7: calculate the wind force using the 
        # relative velocity between the waves and the structure
        F_with_rel_motion[i_] = F_wind(rotor,wind['V_10'], wind['V_hub'][i_], x_dot_hub[i_])

    # Calculate depth and zHub
    z = structure["zBeamNodal"]
    h = np.abs(z[0])
    zHub = z[-1]
    
    outputDict = dict()
    outputDict["t"] = wind["t"]
    outputDict["F"] = F_with_rel_motion
    outputDict["M"] = F_with_rel_motion*(zHub+h)
    
    return outputDict


def calculateStaticWaveLoads(waves, structure, q):
       
    # Calculate the structural velocity along the structural axis
    x_dot = q["alphaDot"][:,None]*structure["phiNodal"][None,:]
    
    # Calculate the structural velocity of the wet part
    wetPart = np.less_equal(structure["zBeamNodal"], 0.)    
    x_dot_submerged = x_dot[:, wetPart]
    zSubmerged = structure["zBeamNodal"][wetPart]
    h = np.abs(zSubmerged[0])
    
    # Initialize the arrays
    F_with_rel_motion, M_with_rel_motion = np.zeros_like(waves["t"]), np.zeros_like(waves["t"])
    
    for i_, t_ in enumerate(waves["t"]):
        u, ut, z = waves["u"][i_,:], waves["ut"][i_,:], waves["z"]
        
        # FIXME Assignment 3 Q1.7: calculate the wave distributed force using the 
        # relative velocity between the waves and the structure        
        #df = np.zeros_like(zSubmerged)
        df = forceDistributed(structure, u, ut, z, x_dot_submerged[i_])
        
        F_with_rel_motion[i_] = np.trapz(df, zSubmerged)
        M_with_rel_motion[i_] = np.trapz(df*(zSubmerged+h), zSubmerged)

    # Output dictionary
    outputDict = dict()
    outputDict["t"] = waves["t"]
    outputDict["F"] = F_with_rel_motion
    outputDict["M"] = M_with_rel_motion
    
    return outputDict    

def calculateDynamicLoads(structure, q):

    # Create output dict
    outputDict = dict()
    outputDict["F"] = np.zeros_like(q["t"])
    outputDict["M"] = np.zeros_like(q["t"])
    
    # some helper quantities
    dz = structure["dz"]
    zElement = structure["zBeamElement"]
    h = np.abs(structure["zBeamNodal"][0])

    # calculations
    outputDict["t"] = q["t"]    
    for i_, t_ in enumerate(q["t"]):
        '''
        # FIXME Assignment 3 Q1.7: insert formula for dynamic loads here
        DistributedForce = structure['rhoAElement']*q['alphaDotDot'][i_]*structure['phiElement']

        outputDict["F"][i_] = np.sum(DistributedForce * dz)

        DistributedMoment = DistributedForce * (zElement + h)
        outputDict["M"][i_] = np.sum(DistributedMoment * dz)
        '''
        outputDict["M"][i_] = np.sum(-structure["rhoAElement"] * dz* q["alphaDotDot"][i_] * structure["phiElement"] * (zElement + h))
        outputDict["F"][i_] = np.sum(-structure["rhoAElement"] * dz* q["alphaDotDot"][i_] * structure["phiElement"])


    return outputDict