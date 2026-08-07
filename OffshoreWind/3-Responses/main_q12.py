'''
Filename: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Code\\classicalSolution\\Stud_\\main_q10.py
Path: c:\\Users\\fabpi\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Courses\\46211_OffshoreWindEnergy\\2024\\Module3\\Code\\classicalSolution\\Stud_
Created Date: Friday, October 18th 2024, 12:36:38 pm
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''

import os
import sys

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

import numpy as np
import pandas as pd
from common import Timer, generateRandomPhases, loadFromJSON
import os
from runner import runEnvironmentalCondition
import matplotlib.pyplot as plt
import rainflow

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)

# In this function, start from what you have in q9 and make a loop
# for all environmental conditions.

# Load the rotor and the monopile which do not change
iea22mw = loadFromJSON(fp("iea22mw.json"))
iea22mw["ARotor"] = 0.25*np.pi*iea22mw["DRotor"]**2
iea22mw["VCutIn"] = 3.0; iea22mw["VCutOut"] = 25.0
monopile = loadFromJSON(fp("monopile.json"))

# Time information
timeInfo = dict()
timeInfo["TDur"] = 3660.
timeInfo["TTrans"] = 60.
timeInfo["dt"] = 0.05
timeInfo["fHighCut"] = 0.5

# Fatigue parameters
mFatigue = 4.
n_eq = 10.**7

# Time factor
# FIXME Assignment 3 Q12: Calculate TLife and Tsim
TLife = 25 * 365 * 24 * 60 * 60                  # 25 years 
TSim = timeInfo["TDur"] - timeInfo["TTrans"]        # seconds in one hour minus transient

# Rescale wind speed by taking into account shear factor 
# FIXME Assignment 3 Q12: include the correct scaleWind parameter using a shear factor of 1/7 and factor 2 on hub height
shear_factor = 1/7
hub_height = 2
scaleWind = hub_height**shear_factor
table40 = pd.read_csv(fp("table40.csv"))
table40["V_10_scaled"] = table40["V_10"]*scaleWind

# Drop values outside of the cut in / cut out
dropStates, = np.where(np.logical_or(table40["V_10_scaled"]<3.0,  table40["V_10_scaled"]>25.0))
table40.drop(dropStates, inplace=True)

# Initialize the M_eq to zero
table40["M_eq"] = 0.

# Loop over ECs
for i_, ec_ in table40.iterrows():
    with Timer(f"Current EC: V={ec_["V_10"]}"):
                                     
        # Here, it is more conveniente to build the dictionaries on the fly
        # rather than reading them from file.
                                               
        # Build wind dict
        wind_ = dict()
        wind_["V_10"] = ec_["V_10_scaled"]
        wind_["l"] = 340.2
        wind_["I"] = ec_["I_norm"] / 100.            
        # Make repeatable random seed, but different between runs
        wind_["randomSeed"] = i_*100 +11

        # Build waves dict
        waves_ = dict()
        waves_["Hs"] = ec_["Hs"]
        waves_["Tp"] = ec_["Tp"]
        waves_["gamma"] = ec_["gamma_fat"]
        waves_["h"] = 34.
        waves_["z"] = np.linspace(-34.0, 0., 35)
        # Make repeatable random seed, but different between runs
        waves_["randomSeed"] = i_*100 +22           
        
        # FIXME Assignment 3 Q12: call the runSeaState function.
        # Then, remove transient and compute the fatigue for each sea state.

        # Run the loads calculation
        Output, response= runEnvironmentalCondition(wind_,
                    waves_,
                    iea22mw,
                    monopile,
                    timeInfo, rotor_state="on")

        # Extract total loads
        outputLoads = Output["total"]

    # Remove transient
    Filter = outputLoads["t"] >= timeInfo["TTrans"]
    outputLoads["t"] = outputLoads["t"][Filter]
    outputLoads["F"] = outputLoads["F"][Filter]
    outputLoads["M"] = outputLoads["M"][Filter]


    # Rainflow count
    # FIXME Assignment 3 Q12: do the rainflow counting of fatigue here (see slides for example)
    rainflowCount = np.array(rainflow.count_cycles(outputLoads['M'])) # apply the rainflow.count_cycles routine
    amplitude = rainflowCount[:,0]/2 # transform range into amplitude
    cycles = rainflowCount[:,1]
    cyclesUpscaled = cycles*TLife/TSim

    # FIXME Assignment 3 Q12: calculate the equivalent moment here 
    # and save it in the dataframe
    table40.loc[i_, "M_eq"] = (np.sum(cyclesUpscaled*amplitude**mFatigue)/n_eq)**(1/mFatigue)
        
# Plot equivalent moment vs wind speed bar chart
plt.figure(figsize=(10,6))
plt.bar(table40["V_10_scaled"], table40["M_eq"], width=0.5,
        color='tab:blue', edgecolor='k', linewidth=0.8)
plt.xlabel(r'Mean Wind Speed $V_{10}$ [m/s]')
plt.ylabel(r'Equivalent Moment $M_{eq}$ [Nm]')
plt.title('Equivalent Moment vs Mean Wind Speed')
plt.show()



# ------------------------------
# Q13
# ------------------------------

# restrict to the currently considered rows (we dropped out-of-range earlier)
probs = table40["p"].astype(float)
probs_sum = probs.sum()

# normalize so probabilities over considered states sum to 1
table40["p_norm"] = probs / probs_sum

# compute the overall life-time equivalent moment using Fatigue exponent mFatigue
# weighted by the normalized state probabilities (damage-equivalence style)
M_eq_states = table40["M_eq"].values
M_eq_global = (np.sum(table40["p"].values * (M_eq_states**mFatigue)))**(1.0/mFatigue)

print(f"Global life-time equivalent moment (25 yr, n_eq={int(n_eq):,}): {M_eq_global/1e6:.3f} MNm")

# ------------------------------
# Q14
# ------------------------------
# damage proxy per state (proportional to damage): p_norm * M_eq^m
damage_raw = table40["p_norm"].values * (table40["M_eq"].values ** mFatigue)

# total damage and relative contribution
damage_total = damage_raw.sum()
damage_fraction = damage_raw / damage_total  # D_state / D_life
table40["D_frac"] = damage_fraction
table40["D_percent"] = damage_fraction * 100.0

# Bar plot: one bar per state (labelled by scaled V_10)
x_labels = table40["V_10_scaled"].round(2).astype(str).values
x_pos = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_pos, table40["D_percent"].values, edgecolor='black', linewidth=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_xlabel(r'Mean Wind Speed $V_{10}$ (scaled) [m/s]')
ax.set_ylabel('Damage contribution [%]')
ax.set_title('Relative contribution of each state to life-time fatigue damage')

# annotate percent above bars
for bar, pct in zip(bars, table40["D_percent"].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()