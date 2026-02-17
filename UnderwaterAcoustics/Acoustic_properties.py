def speed_of_sound_in_water(temperature_celsius, salinity_ppt, depth_meters):
    """Calculate the speed of sound in seawater using the Mackenzie formula."""
    c = (1448.96 +
         4.591 * temperature_celsius -
         0.05304 * temperature_celsius**2 +
         2.374 * 10**-4 * temperature_celsius**3 +
         1.34 * (salinity_ppt - 35) +
         0.0163 * depth_meters + 
         1.675 * 10**-7 * depth_meters**2 -
         0.01025 * temperature_celsius * (salinity_ppt - 35) -
         7.139 * 10**-13 * temperature_celsius * depth_meters**3)
    return c

# Calculations for specific conditions
depths = [10, 40, 70]
speeds = [speed_of_sound_in_water(8, 8, d) for d in depths]

# Print the results
for d, s in zip(depths, speeds):
    print(f"Depth: {d} m, Speed of Sound: {s:.2f} m/s")
