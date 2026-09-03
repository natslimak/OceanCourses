""" Decay Tests """

import numpy as np
import pylab as plt
import os
import sys

# Add the function folder to the path so the helper modules can be imported.
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON, loadConstants
from integration import ode4
from floaterIntegration import dqdt
from plotting import freqSpectrum, makeplots


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
figure_dir = "outputFig"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', output_dir, fname)
get_output_figure = lambda fname: os.path.join(os.path.dirname(__file__), figure_dir, fname)


# ============================================================================
# SETUP: Load other important Files 
# ============================================================================
# Load the constants, timeInfo, and SparBuoyData
constants = loadConstants()
timeInfo = loadFromJSON(get_input_file("time.json"))
SparBuoyData = loadFromJSON(get_output_file("SparBuoyDataComplete.json"))

# Set up the z array for the spar buoy
z = np.linspace(SparBuoyData["z_Bot"], 0., 100)
SparBuoyData["z"] = z

# Load the Rotor info
IEA22MWRotor = loadFromJSON(get_input_file("iea22mw.json"))
IEA22MWRotor["ARotor"] = 0.25 * np.pi * IEA22MWRotor["DRotor"]**2
IEA22MWRotor["gamma"] = 0.              # Controller parameter
IEA22MWRotor["active"] = False          # State of the rotor


# ============================================================================
# Decay Tests
# ============================================================================
#%%  DRY DECAY -> The drag coefficient is set to zero.
#    - no Morison drag force
#    - no viscous damping
#    - only restoring forces remain

# Set the drag coefficient to zero for dry decay
SparBuoyData["CD"] = 0

# Wave kinematics - should be zero
waves = loadFromJSON(get_input_file("nowaves.json"))
waves["z"] = z

# Wind speed - should be zero
wind = loadFromJSON(get_input_file("nowind.json"))
wind.update(timeInfo)
wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
wind["V_hub"] = np.zeros_like(wind["t"])

# calculate the wave kinematics - zero at this stage
waves.update(timeInfo)
waves["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
waves["u"] = np.zeros((len(waves["t"]), len(waves["z"])))
waves["ut"] = np.zeros_like(waves["u"])
waves["eta"] = np.zeros(len(waves["t"]))

# Integration time array
tode = np.arange(0., timeInfo["TDur"], 2*timeInfo["dt"])


# ------ Initial conditions for surge decay -------
q0 = np.array([1,0,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

# Initiate the response dictionary and save the results
response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]

# Calculate the frequency spectrum of the surge response
f, _, S = freqSpectrum(response["t"], response["x5"])
max_S = max(S)
max_f_surge = f[np.argmax(S)]
print(f"Freq Surge: {max_f_surge} Hz")

# Plot the surge response
fig10a = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
fig10a[0,0].figure.savefig(get_output_figure("DryDecay_Surge.pdf"))



# ------ Initial conditions for pitch decay -------
q0 = np.array([0,0.1,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

# Initiate the response dictionary and save the results
response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]

# Calculate the frequency spectrum of the pitch response
f, _, S = freqSpectrum(response["t"], response["x5"])
max_S = max(S)
max_f_pitch = f[np.argmax(S)]
print(f"Freq Pitch: {max_f_pitch} Hz")

# Plot the pitch response
fig10b = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
fig10b[0,0].figure.savefig(get_output_figure("DryDecay_Pitch.pdf"))



#%% Q11: WET DECAY -> The drag coefficient is set to 0.6.
#       - Morison drag force is included
#       - viscous damping is included
#       - restoring forces remain

# Set the drag coefficient to 0.6 for wet decay
SparBuoyData["CD"] = 0.6


# ------ Initial conditions for surge decay -------
q0 = np.array([1,0,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

# Initiate the response dictionary and save the results
response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]

# Plot the surge response for wet decay and dry decay
fig11a = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'g', ax=fig10a)
fig11a[0,0].legend(["no drag", "drag"])
fig11a[0,0].figure.savefig(get_output_figure("Decay_Surge.pdf"))


# ------ Initial conditions for pitch decay -------
q0 = np.array([0,0.1,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode
response["x1"] = q[:,0]
response["x5"] = q[:,1]

# Plot the pitch response for wet decay and dry decay
fig11b = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'g', ax=fig10b)
fig11b[0,0].legend(["no drag", "drag"])
fig11b[0,0].figure.savefig(get_output_figure("Decay_Pitch.pdf"))



#%% Q11 LARGE DECAY TESTS -> with large initial pitch angle = 1 radian (57°)

# ------ Initial conditions for pitch dry decay -------
SparBuoyData["CD"] = 0

q0 = np.array([0,1,0,0,np.nan])
q_nodrag = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_nodrag = dict()
response_nodrag["t"] = tode
response_nodrag["x1"] = q_nodrag[:,0]
response_nodrag["x5"] = q_nodrag[:,1]


# ------ Initial conditions for pitch wet decay -------
SparBuoyData["CD"] = 0.6

q0 = np.array([0,1,0,0,np.nan])
q_drag = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_drag = dict()
response_drag["t"] = tode
response_drag["x1"] = q_drag[:,0]
response_drag["x5"] = q_drag[:,1]


# ------ Plot wet and dry decay test for comparison -------
fig11c = makeplots(wind, waves, SparBuoyData, response_nodrag, timeInfo, 'b')
fig11c = makeplots(wind, waves, SparBuoyData, response_drag, timeInfo, 'g', ax=fig11c)
fig11c[0,0].legend(["no drag", "drag"])

fig11c[0,0].figure.savefig(get_output_figure("Decay_Pitch_Large.pdf"))