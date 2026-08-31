import numpy as np
import os
import sys      

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..','functions', 'python'))
sys.path.append(helpers_path)

from common import loadFromJSON, saveToJSON, loadConstants

inputVariables = "inputVariables"
savedStates = "savedStates"
fp = lambda x: os.path.join(os.path.dirname(__file__),inputVariables,x)
ss = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', savedStates, x)
ov = lambda x: os.path.join(os.path.dirname(__file__), '..', '..', 'outputVariables', x)


# Load the variables here
timeInfo = loadFromJSON(fp("time.json"))
constants = loadConstants()
SparBuoyData = loadFromJSON(fp("SparBuoyData.json"))

g = constants['g']
rhow = constants['rho_water']
rhoa = constants['rho_air']
rhos = SparBuoyData['rho_Steel']
mtu = SparBuoyData['M_Turbine']
zturb = SparBuoyData['z_CM_Turbine']
zhub = SparBuoyData['z_Hub']
fb = SparBuoyData['fb']
draft = SparBuoyData['draft']               # distance from waterline to bottom of "foundation"
Dspar = SparBuoyData['DMonopile']
th = SparBuoyData['Thickness']
Dhp = Dspar
Kmoor = SparBuoyData['K_Moor']
zmoor = SparBuoyData['z_Moor']
Cm = SparBuoyData['CM'] - 1.0
Cd = SparBuoyData['CD']
mt = SparBuoyData['M_Tower']
zCMt = SparBuoyData['z_CM_Tower']
ICMt = SparBuoyData['I_CM_Tower']
BallastHeight = SparBuoyData['BallastHeightindraft']
BallastCOG = SparBuoyData['Ballast_COG']
mb = SparBuoyData['M_Ballast']


IEA22MWRotor = loadFromJSON(fp("iea22mw.json"))

B11 = SparBuoyData["B11"]; 
thrustr = 2*SparBuoyData["MaxThrust"]

##% Preliminary computations
# Calculate center of buoyancy, center of mass, floater inertias
# Location of floater bottom
zbot = -draft

zballst = BallastCOG; # height of draft is ballast 

# FIXME: Center of buoyancy
zCB = -draft/2

# FIXME: Displacement volume (submerged cylinder)
Vol = np.pi*(Dspar/2.0)**2 * draft

# FIXME: Spar length
ls = draft - BallastHeight*draft + fb

# FIXME: Spar mass without ballast (mass of the flaoter)
ms = SparBuoyData['M_Floater']

# Floater mass with ballast
mf = ms + mb

# FIXME: Spar center of mass without ballast
zCMs = SparBuoyData['z_CM_Floater']

# FIXME: Floater center of mass with ballast
zCMf = (ms*zCMs + mb*zballst) / mf

# FIXME: Spar inertia about its Center of Mass without ballast
ICMs = SparBuoyData['I_CM_Floater']


# Distance from floater CM to ballast CM
d_ballast = zballst - zCMs

# Ballast inertia about floater CM (point-mass approximation)
ICMb = mb * d_ballast**2

# FIXME: Floater inertia about floater CM with ballast
ICMf = ICMs + ms * (zCMs - zCMf)**2 + mb * (zballst - zCMf)**2



##% Q6: System matrices

# FIXME: Total mass
mtot = mtu + mt + mb + ms   # floater = spar

# FIXME: Total center of mass
zCMtot = (mtu*zturb + mt*zCMt + ms*zCMs + mb*zballst) / mtot 

# FIXME: Total inertia about flotation point (point O)
Itu_O = mtu * (zturb)**2                # point-mass I - turbine
Itower_O = ICMt + mt * (zCMt)**2        # I for tower about point O
Ispar_O = ICMs + ms * (zCMs)**2         # I for spar/floater about point O
Iballast_O = mb * (zballst)**2          # point-mass I - ballast about point O

IOtot = Itu_O + Itower_O + Ispar_O + Iballast_O

# FIXME: MASS MATRIX
M = np.array([[mtot, mtot*zCMtot],[mtot*zCMtot, IOtot]])


# Ensure arrays
z = np.linspace(zbot, 0, 500)   # bottom to waterline
A_m = np.pi*(Dspar/2)**2
A_m = np.full_like(z, A_m)
a_1 = rhow * Cm * np.trapz(A_m, z)
a_51 = rhow * Cm * np.trapz(z * A_m, z)
a_15 = rhow * Cm * np.trapz(z * A_m, z)
a_5 = rhow * Cm * np.trapz(z**2 * A_m, z)

# FIXME: Added mass matrix
A = np.array([[a_1, a_15],[a_51 ,a_5]])

# DAMPING MATRIX
B = np.array([[B11, 0],[0 ,0]])

# FIXME: Water Plane Inertia
IAA = np.pi/64 * Dspar**4

# FIXME: hydrodynamic stiffness
Chst = np.array([[0, 0], [0, rhow*g*IAA+mtot*g*(zCB-zCMtot)]])

# Mooring restoring matrix
Cmoor = np.array([[Kmoor, Kmoor*zmoor], [Kmoor*zmoor, Kmoor*zmoor**2]])

# RESTORING MATRIX
C = Chst + Cmoor

SparBuoyData["M"] = M
SparBuoyData["C"] = C
SparBuoyData["A"] = A
SparBuoyData["B"] = B

##% Natural Frequencies

# FIXME: calculate C over MA
CoMA=np.matmul(np.linalg.inv(M+A), C)
eigVal, eigVec = np.linalg.eig(CoMA)

# Natural frequencies
omeganat = np.sqrt(eigVal); 
fnat = omeganat/2/np.pi

SparBuoyData["fnat"] = fnat

# Natural periods
Tnat = 1./fnat


"""
# FIXME: Added mass in heave
# A33 = ...; 
a33 = 0.5

# FIXME: Hydrostatic restoring in heave
c33 = 1

# FIXME: Heave natural period
Theave = 1

print(f'Heave period: {Theave:.2f} [s]')
"""


os.makedirs("outputVariables", exist_ok=True)
saveToJSON(SparBuoyData, ov("SparBuoyDataComplete.json"))


print("RESULTS")
print("\n===TASK 1===")
print(f"Total mass of the system: {mtot/1e3:.2f} tonnes")
print(f"Total center of mass of the system: {zCMtot:.2f} m")
print(f"Total inertia about the flotation point: {IOtot/1e6:.2f} x10^6 kg m^2")

print("\n===TASK 7===")
# Display surge and pitch natural periods
print(f'Surge period: {Tnat[0]:.2f} [s]')
print(f'Pitch period: {Tnat[1]:.2f} [s]')
# Display surge and pitch natural frequencies
print("freq_surge = ",fnat[0])
print("freq_pitch = ",fnat[1])
