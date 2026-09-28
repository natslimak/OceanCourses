Um = 0.5                # Mean flow velocity (m/s)
T = 2                   # Wave period (s)
s = 2.65                # Specific gravity of sediment
d = 0.5 * 10**(-3)      # Sediment diameter (m)
D1 = 0.5                # Rock 1 diameter (m)
D2 = 0.05               # Rock 2 diameter (m)
g = 9.81                # Acceleration due to gravity (m/s^2)

KC1 = Um * T / D1
KC2 = Um * T / D2

ratio1 = d/D1
ratio2 = d/D2

# Get the reults
print("KC1:", KC1)
print("KC2:", KC2)
print("d/D1:", ratio1)
print("d/D2:", ratio2)

# Read the mobility number from the graph
# Lecture5, slide 8