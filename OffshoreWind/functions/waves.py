import numpy as np
from common import dispersion, pad2

def calculateRegularWaveParameters(waveDict):
    
    amplitude = np.array([0.5*waveDict["H"]])
    f = np.array([1.0 / waveDict["T"]])
    epsilon=np.array([0.])


    # Store it inside the waves dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["amplitude"] = amplitude
    outputDict["randomPhases"] = epsilon
    outputDict["f"] = f

    return outputDict

def calculateJONSWAPSpectrum(waveDict):
           
    Hs = waveDict["Hs"]
    Tp = waveDict["Tp"]
    
    # If defined, get the gamma. Ohterwise default to 1.0
    gamma = waveDict.get("gamma", 1.0) # Calculated gamma is 1.59 for Tp=13 and Hs=8
                    
    # Calculate frequency information
    df = waveDict["TDur"]**-1.0
    #df = 600**-1.0
    f = np.arange(df, waveDict["fHighCut"], df)
    
    # Spectral width parameter
    sigma = np.ones(len(f))

    fp = 1./Tp

    sigma[f>fp] = 0.09
    sigma[f<=fp] = 0.07        

    # Calculate the Kaimal spectrum
    # Pierson-Moskowitz spectrum
    Spm = 5/16 * Hs**2 * fp**4 * f**(-5) * \
            np.exp( - 5/4 * (f  / fp)**(-4) ) 
    # Jonswap spectrum
    # FIXME Assignment 1 Q6: program the correct spectrum
    Spectrum = 5/16 * Hs**2 * fp**4 * f**(-5) * \
            np.exp( - 5/4 * (f  / fp)**(-4) ) * \
            (1-0.287 * np.log(gamma))* gamma**(np.exp(-0.5 * ((f/fp - 1)/sigma)**2))
    amplitude = np.sqrt(2*Spectrum*df)
    
    #np.random.seed(waveDict["randomSeed"])
    #epsilon=np.array([np.random.uniform(0, 2*np.pi) for _ in range(len(f))])
    # Store it inside the waves dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["Spectrum"] = Spectrum
    outputDict["amplitude"] = amplitude
    outputDict["f"] = f
    #outputDict["randomPhases"] = epsilon
    
    return outputDict
       
def calculateFreeSurfaceElevationTimeSeries(waveDict):
    
    t = waveDict["t"]
    f = waveDict["f"]
    freeSurfTimeSeries = np.zeros_like(t)
    
    
    for i_, _ in enumerate(t):
        for j_, _ in enumerate(f):
            freeSurfTimeSeries[i_] += waveDict["amplitude"][j_]*np.cos(2*np.pi*f[j_]*t[i_] + waveDict["randomPhases"][j_])
    
    # Store the result
    
    outputDict = dict()
    outputDict.update(waveDict)    
    outputDict["t"] = t
    outputDict["eta"] = freeSurfTimeSeries
 
        
    return outputDict

def calculateKinematics(inputDict, wheelerStretching=False):
    #phasesDict=dict()
    #phasesDict=calculateRegularWaveParameters(inputDict)
    t = inputDict["t"]
    f = inputDict["f"]
    omega = 2*np.pi*f
    
    h = inputDict["h"]
    z = inputDict["z"]
    
    u = np.zeros((len(t), len(z)))
    ut = np.zeros((len(t), len(z)))
    
    k = dispersion(f, inputDict["h"])
    
    for i_, _ in enumerate(t):
        for j_, _ in enumerate(z):
            # FIXME Assignment 1 Q2.2
            # Fix velocity calculation
            # Horizontal velocity
            u[i_, j_] = np.sum(inputDict["amplitude"]*omega*np.cosh(k*(z[j_]+h))/np.sinh(k*h)*np.cos(omega*t[i_] + inputDict["randomPhases"]))
        
            # FIXME Assignment 1 Q2.3
            # Code the ut expression
            # Acceleration ut
            ut[i_, j_] = ut[i_, j_] = -np.sum(inputDict["amplitude"] * (omega**2)* (np.cosh(k * (z[j_] + h)) / np.sinh(k * h))* np.sin(omega * t[i_] + inputDict["randomPhases"]))
    
    outputDict = dict()
    outputDict.update(inputDict)        
    outputDict["u"] = u
    outputDict["ut"] = ut
    
    return outputDict

def calculateFreeSurfaceElevationTimeSeriesFFT(waveDict):
    
    t = waveDict["t"]
    f = waveDict["f"]
    
    # FIXME Assignment 3 Q1.8: compute the fft kernel and perform the IFFT
    M = len(t)
    freeSurfTimeSeriesKernel = waveDict["amplitude"] * np.exp(1j * waveDict["randomPhases"]) # compute the freq. domain kernel and pad to M
    freeSurfTimeSeries = M * np.real(np.fft.ifft(pad2(freeSurfTimeSeriesKernel, M))) # perform the IFFT in this line

    # Store the result
    outputDict = dict()
    outputDict.update(waveDict)    
    outputDict["t"] = t
    outputDict["eta"] = freeSurfTimeSeries
        
    return outputDict

def calculateKinematicsFFT(inputDict):
    
    t = inputDict["t"]
    f = inputDict["f"]
    omega = 2*np.pi*f
    
    h = inputDict["h"]
    z = inputDict["z"]
    u = np.zeros((len(t), len(z)))
    ut = np.zeros((len(t), len(z)))
    
    k = dispersion(f, inputDict["h"])
    M = len(t)
    
    for j_, z_ in enumerate(z):
        
        # FIXME Assignment 3 Q1.8: compute the fft kernel and perform the IFFT
        uKernel = inputDict["amplitude"]*omega*np.cosh(k*(z_+h))/np.sinh(k*h) * np.exp(1j * inputDict["randomPhases"]) # compute the freq. domain kernel and pad to M
        u[:, j_] = M * np.real(np.fft.ifft(pad2(uKernel, M))) # perform the IFFT in this line

        utKernel = 1j*inputDict["amplitude"] * (omega**2)* (np.cosh(k * (z_ + h)) / np.sinh(k * h)) * np.exp(1j * inputDict["randomPhases"]) # compute the freq. domain kernel and pad to M
        ut[:, j_] = M * np.real(np.fft.ifft(pad2(utKernel, M))) # perform the IFFT in this line

    outputDict = dict()
    outputDict.update(inputDict)
    outputDict["u"] = u
    outputDict["ut"] = ut
    
    return outputDict

def calculateRegularWaveFrequencyInformation(waveDict):
           
    H = waveDict["Hs"]
    T = waveDict["Tp"]

    
    isRegular = waveDict.get("regular", False)
    if not isRegular:
          raise ValueError("Your input dictionary specifies an irregular sea state, but you have called the regular wave routine.")
                       
    # Calculate frequency information
    
    f = np.array([1./T])
    a = np.array([H/2.])
    
    # Store it inside the wind dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["Spectrum"] = np.nan
    #outputDict["amplitudeSpectrum"] = a
    outputDict["amplitude"] = a
    outputDict["f"] = f
    outputDict["randomPhases"] = np.array([0.])
    
    return outputDict
