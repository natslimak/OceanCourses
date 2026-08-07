import os 
import sys

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from waves import *
from common import *
from monopile import forceIntegrate
import pylab as plt
import os.path
import numpy as np

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
fp = lambda x: os.path.join(inputVariables,x)

# Question 1
# FIXME: input the correct parameters inside wave1.json
wavesQ1 = loadFromJSON(fp("wave1.json"))
    
# Load the time discretization info
timeQ1 = loadFromJSON(fp("time.json"))
wavesQ1.update(timeQ1)

# Calculate the time vector
wavesQ1["t"] = np.arange(0., wavesQ1["TDur"], wavesQ1["dt"])

# Calculate the jonswap spectrum
wavesQ1 = calculateJONSWAPSpectrum(wavesQ1)

randomSeedWaves = 1;
wavesQ1 = generateRandomPhases(wavesQ1, seed=randomSeedWaves)
wavesQ1["etaSlow"] = np.zeros_like(wavesQ1["t"])

# Copy-paste of the slow functions from waves.py
with Timer("etaSlow"):
    for i_, t_ in enumerate(wavesQ1["t"]):
        for j_, _ in enumerate(wavesQ1["f"]):
            wavesQ1["etaSlow"][i_] += wavesQ1["amplitude"][j_]*np.cos(2*np.pi*wavesQ1["f"][j_]*t_ + wavesQ1["randomPhases"][j_])

# FFT solution
fftKernel = wavesQ1["amplitude"]*np.exp(1j*wavesQ1["randomPhases"])
with Timer("etaFast"):
    wavesQ1["etaFast"] = np.real(np.fft.ifft(pad2(fftKernel, len(wavesQ1["t"])))*len(wavesQ1["t"]))


f,ax = plt.subplots(2)
ax[0].plot(wavesQ1["t"], wavesQ1["etaFast"], label="etaFast", alpha=0.5, marker='o')
ax[0].plot(wavesQ1["t"], wavesQ1["etaFast"], label="etaSlow", marker='x', alpha=0.5)
ax[0].set_xlabel("t[s]"); ax[0].set_ylabel("eta[m]")
ax[1].plot(wavesQ1["t"], wavesQ1["etaFast"] - wavesQ1["etaSlow"], label='difference')
ax[0].set_xlabel("t[s]"); ax[0].set_ylabel("delta[m]")
ax[0].legend(); ax[1].legend(); ax[0].grid(); ax[1].grid()
plt.savefig("difference.png")