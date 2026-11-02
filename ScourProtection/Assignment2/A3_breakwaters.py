""" Design Parameters of Breakwaters """

import numpy as np
from scipy.optimize import curve_fit, fsolve, minimize_scalar
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats

# Data from excel files
wave_file = Path(__file__).parent / "Data" / "Hs_Tp_3h_GR8.xls"
df_wave = pd.read_excel(wave_file, sheet_name="Sheet1", header=None, names=["time", "Hs", "Tp"],)

drift_file = Path(__file__).parent / "Data" / "Drift_calculations_results.xlsx"
drift_sheet = pd.read_excel(drift_file, sheet_name=0, header=None)

# Data
L = 50          # Lifetime of the structure (years)
p_f = 0.2       # Exceedance probability of failure


# =================================================================
# 1. POT method to identify extreme wave conditions
# =================================================================

def probability_residual(T_R, p, L):
    """ Calculate the residual of the probability equation. """
    calculated_p = 1 - (1 - 1 / T_R) ** L
    return calculated_p -p

# Solve for the return period T_R 
T_R = fsolve(probability_residual, x0=200, args=(p_f, L),)[0]

# Apply arbitrary time origin to convert hours to datetime
df_wave["time"] = pd.to_datetime(df_wave["time"], unit="h", origin="2000-01-01")
df_wave = df_wave.sort_values("time").reset_index(drop=True)
time_year = (df_wave["time"] - df_wave["time"].iloc[0]).dt.total_seconds() / (24 * 60 * 60* 365)  # Convert to years

# Load the wave data parameters
Hs = df_wave["Hs"].values           # Significant Wave Height
Tp = df_wave["Tp"].values           # Peak Wave Period

# Hand-picked threshold
u = 1.2

# Keep only threshold exceedances
exceedances = df_wave[df_wave["Hs"] > u].copy()

# Start a new storm whenever the gap exceeds 48 hours
time_gap = exceedances["time"].diff()
new_storm = time_gap > pd.Timedelta(hours=48)

# Assign a storm/cluster number
exceedances["storm_id"] = new_storm.cumsum()

# Keep the largest wave from each storm
peaks = (
    exceedances
    .loc[exceedances.groupby("storm_id")["Hs"].idxmax()]
    .sort_values("time")
    .reset_index(drop=True)
)
peaks_time_year = (
    peaks["time"] - df_wave["time"].iloc[0]
).dt.total_seconds() / (24 * 60 * 60 * 365)

print(f"Threshold: {u:.2f} m")
print(f"Number of exceedances: {len(exceedances)}")
print(f"Number of independent storm peaks: {len(peaks)}")



plt.figure(figsize=(14, 5))
plt.plot(time_year, Hs, label="Significant Wave Height (Hs)", color='b')
plt.plot(peaks_time_year, peaks["Hs"], linestyle="None", marker="+", markersize=10,
    markeredgewidth=1.5, color="orange", label="Storm peaks")
plt.axhline(y=u, color='r', linestyle='--')
#plt.title("Significant Wave Height over Time")
plt.xlabel("Time (years)")
plt.ylabel("Significant Wave Height (m)")
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


print(f"Return period: {T_R:.2f} years")



# =================================================================
# 2. Generalized Pareto distribution
# =================================================================

excesses = np.sort((peaks["Hs"] - u).to_numpy())[::-1]
n = len(excesses)
i = np.arange(1, n + 1)
plotting_position = 1 - i / (n + 1)


def gpd_cdf_excess(excess, ksi, sigma):
    """GPD CDF from the course equation, using excess = Hs - u."""
    return 1 - (1 - (ksi * excess) / sigma) ** (1 / ksi)

# Fit the GDP curve to the plotting_position data
gpd_parameters, _ = curve_fit(gpd_cdf_excess, excesses, plotting_position,
    p0=(0.1, 1.0), bounds=((1e-6, 1e-6), (0.9, np.inf)), maxfev=10000,)
gpd_ksi, gpd_sigma = gpd_parameters


plt.plot(excesses, plotting_position, "+", label="Storm peaks")
fitted_excesses = np.linspace(0, excesses.max(), 100)
plt.plot(fitted_excesses, gpd_cdf_excess(fitted_excesses, gpd_ksi, gpd_sigma), "r-", label="GPD fitted curve")
plt.title("GPD fitted directly to plotting positions")
plt.xlabel(r"$X = H_s-u$ [m]")
plt.ylabel(r"$F$")
plt.grid()
plt.legend()
plt.show()

print(f"GPD shape (ksi): {gpd_ksi:.4f}")
print(f"GPD scale (sigma): {gpd_sigma:.4f} m")


y = excesses * (gpd_ksi / gpd_sigma)
x = excesses

# Fit a line to x and y 
fit = np.polyfit(x, y, 1)
slope, intercept = fit

plt.plot(x, y, "+", label="Storm peaks")
plt.plot(x, slope * x + intercept, "r-", label="Fitted line")
plt.title("GPD fitted to transformed data")
plt.xlabel(r"$X = H_s-u$ [m]")
plt.ylabel(r"$Y = \xi X / \sigma$")
plt.grid()
plt.show()


# =================================================================
# 3. Relationship between the Hs and Tp
# =================================================================

"""
Tp_est = 3.4 * np.sqrt(df_wave["Hs"])  # Empirical relationship between Hs and Tp

plt.figure(figsize=(10, 6))
plt.scatter(df_wave["Hs"], df_wave["Tp"], color='b', label="Storm Peaks")
plt.plot(df_wave["Hs"], Tp_est, color='r', label="Estimated Tp (3.4 * sqrt(Hs))")
plt.title("Relationship between Significant Wave Height (Hs) and Peak Wave Period (Tp)")
plt.xlabel("Significant Wave Height (Hs) [m]")
plt.ylabel("Peak Wave Period (Tp) [s]")
plt.grid()
plt.legend()
plt.show()"""

# =================================================================
# 4. Estimate the design Hs and Tp
# =================================================================
