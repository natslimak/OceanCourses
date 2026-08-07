import numpy as np
from common import pad2

def calculateKaimalSpectrum(windDict):
    
    # Store it inside the wind dictionary
    outputDict = dict()
    outputDict.update(windDict)
    
    # Calculate frequency information
    df = windDict["TDur"]**-1
    f = np.arange(df, windDict["fHighCut"], df)

    # Calculate the Kaimal spectrum
    # FIXME Assignment 3 Q1.3: Program spectrum 
    Spectrum = (4 * windDict["I"]**2 * windDict["V_10"] * windDict["l"])/(1 + (6 * f * windDict["l"] / windDict["V_10"]))**(5/3)
    amplitudeSpectrum = np.sqrt(2*Spectrum*df)

    #np.random.seed(windDict["randomSeed"])
    #epsilon=np.array([np.random.uniform(0, 2*np.pi) for _ in range(len(f))])
    

    outputDict["Spectrum"] = Spectrum
    outputDict["amplitudeSpectrum"] = amplitudeSpectrum
    outputDict["f"] = f
    #outputDict["randomPhases"] = epsilon
    
    return outputDict

def calculateWindTimeSeries(windDict):
    t = windDict["t"]
    f = windDict["f"]
    windTimeSeries = np.zeros_like(t)
    
    for i_, _ in enumerate(t):
        for j_, _ in enumerate(f):
            # FIXME Assignment 3 Q1.3: add random phases
            # Hint: same as you did in waves.py in assignment 1
            windTimeSeries[i_] += windDict["amplitudeSpectrum"][j_]*np.cos(2*np.pi*f[j_]*t[i_] + windDict["randomPhases"][j_]) 
    
    # Store the result
    outputDict = dict()
    outputDict.update(windDict)
    outputDict["t"] = t
    outputDict["V_hub"] = windTimeSeries + windDict["V_10"]
    
    return outputDict


def calculateWindTimeSeriesFFT(windDict):
    t = windDict["t"]
    f = windDict["f"]
    windTimeSeries = np.zeros_like(t)
    
    M = len(t)
    # FIXME Assignment 3 Q1.8: compute the fft kernel and perform the IFFT
    windTimeSeriesKernel = windDict["amplitudeSpectrum"] * np.exp(1j * windDict["randomPhases"]) # compute the freq. domain kernel and pad to M
    windTimeSeries = M * np.real(np.fft.ifft(pad2(windTimeSeriesKernel, M))) # perform the IFFT in this line
    
    # Store the result
    outputDict = dict()
    outputDict.update(windDict)
    outputDict["t"] = t
    outputDict["V_hub"] = windTimeSeries + windDict["V_10"]
    
    return outputDict