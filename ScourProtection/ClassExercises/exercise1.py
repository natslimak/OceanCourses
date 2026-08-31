import matplotlib.pyplot as plt
import numpy as np

h = 0.5         # Water depth [m]
V = 0.45        # Depth-averaged current velocity [m/s]
visc = 1.0e-6   # Kinematic viscosity of water [m^2/s]
d = 0.3e-3      # Grain diameter [m]
ks = 2.5 * d    # Roughness height [m]
s = 2.65        # Relative density of sediment [-]
kappa = 0.4     # von Karman constant [-]
g = 9.81        # Gravitational acceleration [m/s^2]



# === Calcualte the Shields Parameter ===

# Get the U_friction velocity
U_f = V / (6 + 1/kappa) * np.log(h/ks)

# Calculate the Shields parameter
theta = U_f**2 / ((s-1) * g * d)

# Incipient Motion of Sediment Grains
Re = U_f * d / visc

# Get the results
print(f"Friction velocity: {U_f:.4f} m/s")
print(f"Shields parameter: {theta:.4f}")
print(f"Reynolds number: {Re:.4f}")

# Plot the Shields curve
plt.figure(figsize=(8, 6))
plt.plot(theta, Re, 'o', label='Incipient Motion')
plt.xlabel('Shields Parameter')
plt.ylabel('Reynolds Number')
plt.title('Sediment Motion')
plt.legend()
plt.grid(True)
plt.show()
