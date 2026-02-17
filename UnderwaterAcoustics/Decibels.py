import math


""" TASK 1 """
speed_of_sound_baltic = 1495  # Speed of sound in Baltic Sea water in m/s

def calculate_wavelength(frequency_khz, speed_of_sound_ms):
    """Calculate the wavelength of a sound wave in water."""
    frequency_hz = frequency_khz * 1000  # Convert kHz to Hz
    wavelength_m = speed_of_sound_ms / frequency_hz
    return wavelength_m

frequencies_khz = [120, 38]

# Print the results
print("TASK 1:")
for d, s in zip(frequencies_khz, [calculate_wavelength(f, speed_of_sound_baltic) for f in frequencies_khz]):
    print(f"Frequency: {d} kHz, Wavelength: {s * 100:.2f} centimeters")




""" TASK 2 """
rho_baltic = 1025        # Density of Baltic Sea water in kg/m^3
p_source = 200*10**3     # Sound pressure in Pa
p_ref = 1e-6             # Reference sound pressure in µPa

# dB = 10 * log10(I / I_ref) = 20 * log10(P / P_ref)
dB = 20 * math.log10(p_source / p_ref)  

print("\nTASK 2:")
print(f"Decibel Level: {dB:.2f} dB")




""" TASK 3 """
# Decibels reduced by 6 dB 
new_dB = dB - 6

# solve for new pressure level p_source_new = p_ref * 10**(new_dB / 20)
p_source_new = p_ref * 10**(new_dB / 20)


print("\nTASK 3:")
print(f"New Pressure Level: {p_source_new / 1000:.2f} kPa")
print(f"Ratio of new pressure to original pressure: {p_source_new / p_source:.2f}")





""" TASK 4 """
import sympy as sp
I_0, I_1 = sp.symbols('I_0 I_1')

sound_intensity = 0.001 / 100       # Sound intensity in procent
I_1 = I_0 * sound_intensity         # I_1 is the reflected intensity, I_0 is the emitted intensity


# Calculate the energy flux in W/m^2
I = 10 * math.log10(I_0/I_1)  # Convert promille to W/m^2 and calculate dB

print("\nTASK 4:")
print(f"Change in dB: {I:.2f} dB")





""" TASK 5 """
D = 20         # Depth in meters
S = 8          # Salinity in ppt
T = 10         # Temperature in Celsius



print("\nTASK 5:")