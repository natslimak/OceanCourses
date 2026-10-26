"""" Shoreline Properties """

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data 
wave_file = Path(__file__).parent / "Data" / "Hs_Tp_3h_GR8.xls"
df_wave = pd.read_excel(wave_file, sheet_name="Sheet1", header=None, names=["time", "Hs", "Tp"],)

drift_file = Path(__file__).parent / "Data" / "Drift_calculations_results.xlsx"
drift_sheet = pd.read_excel(drift_file, sheet_name=0, header=None)


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

print(f"\nClosure Depth: {d_l:.3f} m")



# =================================================================
# 1.3 Sediment Q-𝛼 curve 
# =================================================================

# Load from excel sheet
distance = drift_sheet.iloc[22, 2:].dropna().to_numpy(dtype=float)
orientations = drift_sheet.iloc[4:21:2, 0].to_numpy(dtype=float)
transport_columns = slice(2, 2 + len(distance))
right_drift = drift_sheet.iloc[4:21:2, transport_columns].to_numpy(dtype=float)
left_drift = drift_sheet.iloc[5:22:2, transport_columns].to_numpy(dtype=float)

# Get the total transport rates
right_transport = np.sum(right_drift, axis=1)
left_transport = np.sum(left_drift, axis=1)

net_transport = right_transport + left_transport
gross_transport = right_transport - left_transport

q_alpha = pd.DataFrame(
    {
        "alpha_deg": orientations,
        "net_m3_per_year": net_transport,
        "gross_m3_per_year": gross_transport,
        "left_m3_per_year": left_transport,
        "right_m3_per_year": right_transport,
    }
)

print("\nQ-alpha transport results:")
print(q_alpha.to_string(index=False, float_format="{:.3f}".format))

plt.figure()
plt.plot(orientations, net_transport, "o-", label="Net")
plt.plot(orientations, gross_transport, "o-", label="Gross")
plt.plot(orientations, left_transport, "o-", label="Left")
plt.plot(orientations, right_transport, "o-", label="Right")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel(r"Shoreline orientation, $\alpha$ [$^\circ$]")
plt.ylabel(r"Transport, $Q$ [$\mathrm{m}^3/\mathrm{m}/\mathrm{year}$]")
plt.title(r"Sediment transport $Q$-$\alpha$ curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =================================================================
# 1.4 Null Net Orientation
# =================================================================

# Interpolate between the points to find the null net orientation.
# A null orientation occurs where net transport changes sign.
exact_zero = np.flatnonzero(np.isclose(net_transport, 0))
sign_changes = np.flatnonzero(net_transport[:-1] * net_transport[1:] < 0)

if exact_zero.size:
    null_net_orientation = orientations[exact_zero[0]]
elif sign_changes.size:
    i = sign_changes[0]
    null_net_orientation = orientations[i] - (
        net_transport[i]
        * (orientations[i + 1] - orientations[i])
        / (net_transport[i + 1] - net_transport[i])
    )
else:
    raise ValueError("No null net orientation found in the supplied data.")

print(f"\nNull Net Orientation: {null_net_orientation:.3f} degrees")
