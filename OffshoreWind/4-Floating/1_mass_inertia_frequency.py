""" Calculate the system matrices for the spar buoy and compute the natural frequencies of the system. """

import numpy as np
import os
import sys      

# Add functions folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions'))
sys.path.append(helpers_path)

from common import loadFromJSON, saveToJSON, loadConstants


# ============================================================================
# SETUP: Load Input Files
# ============================================================================
input_dir = "inputVariables"
output_dir = "savedStates"
get_input_file = lambda fname: os.path.join(os.path.dirname(__file__), input_dir, fname)
get_output_file = lambda fname: os.path.join(os.path.dirname(__file__), '..', '..', output_dir, fname)

# Load the variables here
timeInfo = loadFromJSON(get_input_file("time.json"))
constants = loadConstants()
SparBuoyData = loadFromJSON(get_input_file("SparBuoyData.json"))
IEA22MWRotor = loadFromJSON(get_input_file("iea22mw.json"))


# ============================================================================
# Calculate the system matrices for the spar buoy
# ============================================================================
zbot = -SparBuoyData['draft']           # Location of floater bottom

zballst = SparBuoyData['Ballast_COG']   # Ballast center of gravity in the draft direction

# Center of buoyancy for a uniformly submerged cylinder is at half the draft
zCB = -SparBuoyData['draft'] / 2.0

# Displacement volume of the submerged cylindrical spar
Vol = np.pi * (SparBuoyData['DMonopile'] / 2.0) ** 2 * SparBuoyData['draft']

# Spar length from waterline to the top of the body
ls = SparBuoyData['draft'] - SparBuoyData['BallastHeightindraft'] * SparBuoyData['draft'] + SparBuoyData['fb']

# Floater mass with ballast
mf = SparBuoyData['M_Floater'] + SparBuoyData['M_Ballast']

# Floater center of mass with ballast included
zCMf = (SparBuoyData['M_Floater'] * SparBuoyData['z_CM_Floater'] + SparBuoyData['M_Ballast'] * zballst) / mf

# Distance from floater CM to ballast CM
d_ballast = zballst - SparBuoyData['z_CM_Floater']

# Ballast inertia about the spar CM using point-mass approximation
ICMb = SparBuoyData['M_Ballast'] * d_ballast**2

# Floater inertia about its combined center of mass
ICMf = (
      SparBuoyData['I_CM_Floater']
    + SparBuoyData['M_Floater'] * (SparBuoyData['z_CM_Floater'] - zCMf) ** 2
    + SparBuoyData['M_Ballast'] * (zballst - zCMf) ** 2
)


# ============================================================================
# System Matrices
# ============================================================================

# Total mass of the complete system
mtot = (
      SparBuoyData['M_Turbine']
    + SparBuoyData['M_Tower']
    + SparBuoyData['M_Ballast']
    + SparBuoyData['M_Floater']
)

# Total center of mass of the system
zCMtot = (
      SparBuoyData['M_Turbine'] * SparBuoyData['z_CM_Turbine']
    + SparBuoyData['M_Tower'] * SparBuoyData['z_CM_Tower']
    + SparBuoyData['M_Floater'] * SparBuoyData['z_CM_Floater']
    + SparBuoyData['M_Ballast'] * zballst
) / mtot

# Inertia contributions about the flotation point (origin)
Itu_O = SparBuoyData['M_Turbine'] * SparBuoyData['z_CM_Turbine'] ** 2
Itower_O = SparBuoyData['I_CM_Tower'] + SparBuoyData['M_Tower'] * SparBuoyData['z_CM_Tower'] ** 2
Ispar_O = SparBuoyData['I_CM_Floater'] + SparBuoyData['M_Floater'] * SparBuoyData['z_CM_Floater'] ** 2
Iballast_O = SparBuoyData['M_Ballast'] * zballst ** 2

IOtot = Itu_O + Itower_O + Ispar_O + Iballast_O


# Mass matrix for surge and pitch coupling
M = np.array([[mtot, mtot * zCMtot], [mtot * zCMtot, IOtot]])


# Ensure arrays
z = np.linspace(zbot, 0, 500)
A_m = np.pi * (SparBuoyData['DMonopile'] / 2.0) ** 2
A_m = np.full_like(z, A_m)

# Added mass coefficients (use np.trapezoid instead of nonexistent np.trapezoid)
a_1 = constants['rho_water'] * (SparBuoyData['CM'] - 1.0) * np.trapezoid(A_m, z)
a_51 = constants['rho_water'] * (SparBuoyData['CM'] - 1.0) * np.trapezoid(z * A_m, z)
a_15 = a_51
a_5 = constants['rho_water'] * (SparBuoyData['CM'] - 1.0) * np.trapezoid(z**2 * A_m, z)

A = np.array([[a_1, a_15], [a_51, a_5]])

# Damping matrix for surge motion
B = np.array([[SparBuoyData['B11'], 0], [0, 0]])

# Waterplane inertia for the spar section
IAA = np.pi / 64.0 * SparBuoyData['DMonopile'] ** 4

# Hydrostatic restoring around pitch plus gravity offset from buoyancy and mass centers
Chst = np.array(
    [
        [0, 0],
        [0, constants['rho_water'] * constants['g'] * IAA + mtot * constants['g'] * (zCB - zCMtot)],
    ]
)

# Mooring stiffness matrix for surge and pitch coupling
Cmoor = np.array(
    [
        [SparBuoyData['K_Moor'], SparBuoyData['K_Moor'] * SparBuoyData['z_Moor']],
        [SparBuoyData['K_Moor'] * SparBuoyData['z_Moor'], SparBuoyData['K_Moor'] * SparBuoyData['z_Moor'] ** 2],
    ]
)

# Total restoring matrix
C = Chst + Cmoor

SparBuoyData['M'] = M
SparBuoyData['C'] = C
SparBuoyData['A'] = A
SparBuoyData['B'] = B


# ============================================================================
# Natural Frequencies of the System
# ============================================================================

# Generalized eigenvalue problem for surge and pitch natural modes
CoMA = np.matmul(np.linalg.inv(M + A), C)
eigVal, eigVec = np.linalg.eig(CoMA)


# If eigenvalues have tiny imaginary parts due to numerical error, drop them;
# if they are negative, take absolute value before sqrt to avoid complex results.
imag_mag = np.max(np.abs(np.imag(eigVal)))
eigVal_real = np.real_if_close(eigVal, tol=1e-8)
eigVal_real = np.abs(np.real(eigVal_real))

omeganat = np.sqrt(eigVal_real)
fnat = omeganat / (2 * np.pi)

# Store as plain Python lists so saveToJSON can serialize them
SparBuoyData["fnat"] = fnat.tolist()

# Natural periods (also serializable)
Tnat = (1.0 / fnat).tolist()
SparBuoyData["Tnat"] = Tnat



# ============================================================================
# Save the results and get statistics
# ============================================================================

# Ensure the output directory exists (use the savedStates output path)
os.makedirs(os.path.dirname(get_output_file("SparBuoyDataComplete.json")), exist_ok=True)
saveToJSON(SparBuoyData, get_output_file("SparBuoyDataComplete.json"))

print("RESULTS")

# Mass, center of mass and inertia
print(f"Total mass of the system: {mtot/1e3:.2f} tonnes")
print(f"Total center of mass of the system: {zCMtot:.2f} m")
print(f"Total inertia about the flotation point: {IOtot/1e6:.2f} x10^6 kg m^2")

# Display surge and pitch natural periods
print(f'Surge period: {Tnat[0]:.2f} [s]')
print(f'Pitch period: {Tnat[1]:.2f} [s]')

# Display surge and pitch natural frequencies
# Ensure frequencies are real scalars in Hz and print with units
freq_surge = float(np.real(fnat[0]))
freq_pitch = float(np.real(fnat[1]))
print(f'freq_surge = {freq_surge:.6f} Hz')
print(f'freq_pitch = {freq_pitch:.6f} Hz')
