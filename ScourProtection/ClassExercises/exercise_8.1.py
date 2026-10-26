""" Calculate the volume of sediment transport in the profile. """

delta_A = 4000000           # Area of the sediment transport section [m^2]
h_p = 12                    # Height of the profile in northjylland [m]
delta_t = 60                # Time period for the sediment transport [year]
delta_Q = delta_A * h_p

# Calculate the sediment transport rate in m^3/s
sediment_transport = delta_Q / delta_t
print(f"\nSediment Transport: {sediment_transport:.3f} m^3/year")