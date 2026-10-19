"""" Shoreline Properties """

from pathlib import Path
import numpy as np
import pandas as pd


# Data 
wave_file = Path(__file__).parent / "Data" / "Hs_Tp_3h_GR9.xls"
df_wave = pd.read_excel(wave_file, sheet_name="Sheet1", header=None, names=["time", "Hs", "Tp"],)

sediment_file = Path(__file__).parent / "Data" / "Sediment_Grain_Size.xlsx"
df_sediment = pd.read_excel(sediment_file, sheet_name="Sheet1", header=None, names=["time", "D50", "D90"],)


# =================================================================
# 1.1 Significant Wave Height exceeeding 12 hours per year
# =================================================================

Hs = df_wave["Hs"].values           # Significant Wave Height
Tp = df_wave["Tp"].values           # Peak Wave Period

hours_per_year = 365 * 24           # Total hours in a year
hours_exceeded_per_year = 12        # Maximum hours per year that can be exceeded

# Loop through critical Hs values to find the threshold where exceedance is below the allowed limit
for critical_Hs in np.arange(0.1, 100, 0.01):
    exceedance = np.sum(Hs > critical_Hs) / len(Hs)
    if exceedance < (hours_exceeded_per_year / hours_per_year):
        break

# Get the wave period for that critical Hs value
critical_Tp = np.mean(Tp[Hs > critical_Hs])

# Print the results
print(f"Exceeded Hs: {critical_Hs:.3f} m")
print(f"Exceeded Tp: {critical_Tp:.3f} s")



# =================================================================
# 1.2 Closure Depth
# =================================================================

# Hallermeiers formula for assessing closure depth
d_l = 1.57 * critical_Hs * (critical_Tp / 10) ** (1/3)

print(f"Closure Depth: {d_l:.3f} m")



# =================================================================
# 1.3 Sediment Q-𝛼 curve 
# =================================================================

