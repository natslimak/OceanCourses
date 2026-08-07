""" Thrust force calculation for a wind turbine rotor. """

import os
import sys 
import numpy as np
import pylab as plt

# Add the functions folder to the path 
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON
from rotor import F_avg


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)

# Load wind turbine and wind properties (rotor diameter, wind speed, etc.)
iea22mw = loadFromJSON(get_input_file("iea22mw.json"))
wind_data = loadFromJSON(get_input_file("wind.json"))
time_data = loadFromJSON(get_input_file("time.json"))


# ============================================================================
# Compute thrust force for a range of wind speeds
# ============================================================================
thrust = dict()
thrust["V"] = np.arange(0., 25.)
thrust["T"] = np.zeros_like(thrust["V"])

for i_, V_ in enumerate(thrust["V"]):
    thrust["T"][i_] = F_avg(iea22mw, V_)


# ============================================================================
# Plot: Thrust force as a function of wind speed
# ============================================================================
plt.plot(thrust["V"], thrust["T"])
plt.xlabel('Wind speed V [m/s]')
plt.ylabel('Thrust force T [N]')
plt.grid()
plt.title('Thrust force in function of the wind speed')
plt.show()

