import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so 'acoustics' package can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoustics.gsw_sound_speed import gsw_sound_speed
from acoustics.SS_CTD import mackenzie_sound

# Load oceanographic data
data_dir = Path(__file__).parent.parent / "data"
ds = xr.load_dataset(data_dir / "NA_TS_P1D-d230605.nc")

# Extract temperature and salinity profiles at a specific location
lat0, lon0 = 52.0, -35.9  # example location
T = ds.thetao.isel(time=0).sel(latitude=lat0, longitude=lon0, method="nearest")
S = ds.so.isel(time=0).sel(latitude=lat0, longitude=lon0, method="nearest")
depth = ds.depth

# Convert depth to pressure (approximate: 1 dbar ≈ 1 meter depth)
pressure = depth.values

# Compute sound speed using both methods
# GSW expects arrays, so convert profiles to lists
gsw_ss = gsw_sound_speed(S.values, T.values, pressure)
mackenzie_ss = mackenzie_sound(T.values, S.values, depth.values)

# Calculate relative error (%)
relative_error = (mackenzie_ss - gsw_ss) / gsw_ss * 100

# Display results
print("Depth (m) | GSW (m/s) | Mackenzie (m/s) | Relative Error (%)")
print("-" * 65)
for i in range(min(10, len(depth))):  # Show first 10 depths
    print(f"{depth.values[i]:8.1f} | {gsw_ss[i]:9.2f} | {mackenzie_ss[i]:15.2f} | {relative_error[i]:18.4f}")

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharey=True)

ax[0].plot(gsw_ss, depth, label='GSW', linewidth=2)
ax[0].plot(mackenzie_ss, depth, label='Mackenzie', linewidth=2, linestyle='--')
ax[0].set_xlabel('Sound Speed (m/s)', fontsize=12)
ax[0].set_ylabel('Depth (m)', fontsize=12)
ax[0].legend()
ax[0].invert_yaxis()
ax[0].grid(True, alpha=0.3)
ax[0].set_title('Sound Speed Comparison')

ax[1].plot(relative_error, depth, linewidth=2, color='red')
ax[1].set_xlabel('Relative Error (%)', fontsize=12)
ax[1].grid(True, alpha=0.3)
ax[1].set_title('Mackenzie vs TEOS-10')

ax[2].plot(T, depth, linewidth=2, color='orange', label='Temperature')
ax[2].set_xlabel('Temperature (°C)', fontsize=12)
ax[2].grid(True, alpha=0.3)
ax[2].legend()
ax[2].set_title('Temperature Profile')

plt.tight_layout()
plt.show()

""" EXPLANATION:
At warmer temperatures near the surface, the non-linear interactions between temperature, 
salinity, and pressure become more significant. Mackenzie's empirical formula (from 1981) 
doesn't capture these non-linearities as accurately as the modern TEOS-10 standard."""