""" Design Parameters of Breakwaters """

import numpy as np
from scipy.optimize import curve_fit, fsolve, minimize_scalar
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats
from scipy.stats import genpareto


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

########################################################################################
# --------------------Cool ChatGPT diagnostics--------------------------------------
########################################################################################

# Mean Residual Life analysis

# Establish a fixed catalogue of storm peaks
u_base = 0.5

events = df_wave.loc[
    df_wave["Hs"] > u_base, ["time", "Hs"]
].copy()

new_storm = events["time"].diff() > pd.Timedelta(hours=48)
events["storm_id"] = new_storm.cumsum()

storm_peaks = events.groupby("storm_id")["Hs"].max().to_numpy()

# Candidate thresholds
thresholds = np.arange(0.6, 1.601, 0.025)

mean_excess = []
n_peaks = []

for threshold in thresholds:
    selected = storm_peaks[storm_peaks > threshold]
    n_peaks.append(len(selected))

    mean_excess.append(
        np.mean(selected - threshold)
        if len(selected) > 0 else np.nan
    )

mean_excess = np.array(mean_excess)
n_peaks = np.array(n_peaks)

# Plot MRL
fig, ax = plt.subplots(figsize=(10, 6))

sufficient = n_peaks >= 20

ax.plot(thresholds[sufficient], mean_excess[sufficient],
        "o-", label="At least 20 storm peaks")

ax.plot(thresholds[~sufficient], mean_excess[~sufficient],
        "x", color="red", label="Fewer than 20 peaks")

ax.axvline(1.2, linestyle="--", color="orange",
           label="u = 1.2 m")
ax.axvline(1.4, linestyle="--", color="green",
           label="u = 1.4 m")

ax.set_xlabel("Threshold u [m]")
ax.set_ylabel("Mean excess [m]")
ax.grid()
ax.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(thresholds, n_peaks, ":", color="grey")
ax2.set_ylabel("Number of independent storm peaks")

plt.title("Mean Residual Life: Independent Storm Peaks")
plt.tight_layout()
plt.show()

# Compare thresholds
for threshold in [1.2, 1.4]:
    selected = storm_peaks[storm_peaks > threshold]
    print(f"\nThreshold: {threshold:.2f} m")
    print(f"Number of peaks: {len(selected)}")
    if len(selected):
        print(f"Mean excess: {np.mean(selected - threshold):.4f} m")

########################################################################################
#--------------------------------- Cool ChatGPT diagnostics------------------------------
########################################################################################


# Hand-picked threshold
u = 0.8

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

# Initiate the plotting positions for the excesses (Hs - u)
excesses = np.sort((peaks["Hs"] - u).to_numpy())[::-1]
n = len(excesses)
i = np.arange(1, n + 1)
plotting_position = 1 - i / (n + 1)     # based on distribution of frequencie


def gpd_cdf(excess, ksi, sigma):
    """ GPD CDF using the presentation convention"""
    value = 1 - ksi * excess / sigma
    return np.where(value > 0, 1 - value ** (1 / ksi), 1.0)


# Least-squares fit to the empirical CDF.
ls_parameters, _ = curve_fit(
    gpd_cdf,
    excesses,
    plotting_position,
    p0=(0.1, 1.0),
    bounds=((-0.9, 1e-6), (0.49, np.inf)),
    maxfev=10000,
)
ls_ksi, ls_sigma = ls_parameters

# Method-of-moments estimates for a GPD:
# mean = sigma / (1 - ksi), variance = sigma**2 / ((1 - ksi)**2 * (1 - 2*ksi)).
excess_mean = excesses.mean()
excess_variance = excesses.var(ddof=0)
mom_ksi = -0.5 * (1 - excess_mean**2 / excess_variance)
mom_sigma = excess_mean * (1 + mom_ksi)




plt.plot(excesses, plotting_position, "+", label="Storm peaks")
fitted_excesses = np.linspace(0, excesses.max(), 100)
plt.plot(fitted_excesses, gpd_cdf(fitted_excesses, ls_ksi, ls_sigma),
    "r-", label="Least-squares fit")
plt.plot(fitted_excesses, gpd_cdf(fitted_excesses, mom_ksi, mom_sigma),
    "g-", label="Method of moments")
#plt.plot(fitted_excesses, gpd_cdf(fitted_excesses, ml_ksi, ml_sigma),
    #"b:", label="Maximum likelihood")
plt.title("GPD fitted directly to plotting positions")
plt.xlabel(r"$X = H_s-u$ [m]")
plt.ylabel(r"$F$")
plt.grid()
plt.legend()
plt.show()

print(f"Least-squares:       ksi = {ls_ksi:.4f}, sigma = {ls_sigma:.4f} m")
print(f"Method of moments:   ksi = {mom_ksi:.4f}, sigma = {mom_sigma:.4f} m")


# Return levels use the annual rate of independent storm peaks above u.
record_years = time_year.iloc[-1]
peaks_per_year = len(peaks) / record_years
return_periods = np.array([1, 5, 10, 25, 50, 100, T_R])


def return_level(return_period, ksi, sigma):
    """Estimate Hs exceeded once every return_period years."""
    rate_term = peaks_per_year * return_period

    if np.isclose(ksi, 0):
        excess = sigma * np.log(rate_term)
    else:
        excess = sigma / ksi * (1 - rate_term ** (-ksi))

    return u + excess

plt.figure()
plt.plot(return_periods, return_level(return_periods, ls_ksi, ls_sigma),
    "o-", label="Least squares")
plt.plot(return_periods, return_level(return_periods, mom_ksi, mom_sigma),
    "s--", label="Method of moments")
plt.axvline(T_R, color="k", linestyle=":", label=f"Required T = {T_R:.1f} years")
plt.xlabel("Return period [years]")
plt.ylabel(r"Estimated wave height $H_s$ [m]")
plt.xscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Select the parameter set after inspecting the return-level plot.
selected_method = "least_squares"
parameter_sets = {
    "least_squares": (ls_ksi, ls_sigma),
    "method_of_moments": (mom_ksi, mom_sigma),
}
selected_ksi, selected_sigma = parameter_sets[selected_method]
design_hs = return_level(T_R, selected_ksi, selected_sigma)
print(f"Selected method: {selected_method}")
print(f"Design Hs for T = {T_R:.2f} years: {design_hs:.3f} m")


#y = excesses * (selected_ksi / selected_sigma)
# Answer: ksi 0.0327, sigma 0.2


x = excesses
F = plotting_position

ksi_range = np.linspace(0.0001, 0.5, 5000)

smallest_error = np.inf
best_ksi = None
best_sigma = None
errors = []

for ksi in ksi_range:

    # Reduced variate from the lecture
    y = 1 - (1 - F) ** ksi

    # Least-squares regression through the origin
    A = np.sum(x * y) / np.sum(x ** 2)

    # Calculate sigma
    sigma = ksi / A

    # Check that the distribution is valid
    if np.any(1 - ksi * x / sigma <= 0):
        errors.append(np.nan)
        continue

    # Estimated wave heights from inverse GPD
    x_estimated = sigma / ksi * y

    # Goodness-of-fit based on actual wave heights
    Hs_estimated = u + x_estimated
    Hs_observed = u + x

    error = np.mean(
        np.abs(Hs_estimated - Hs_observed) / Hs_observed
    )

    errors.append(error)

    if error < smallest_error:
        smallest_error = error
        best_ksi = ksi
        best_sigma = sigma

print("Results of the parameter search:")
print(f"Smallest error: {smallest_error:.4%}")
print(f"ksi: {best_ksi:.5f}")
print(f"sigma: {best_sigma:.5f}")

# Plot goodness-of-fit versus ksi
plt.figure()
plt.plot(ksi_range, errors)
plt.axvline(best_ksi, color="r", linestyle="--",
            label=f"Best ksi = {best_ksi:.4f}")
plt.xlabel(r"$\xi$")
plt.ylabel("Average relative error")
plt.title("GPD goodness-of-fit")
plt.grid()
plt.legend()
plt.show()

# Plot transformed least-squares fit
y_best = 1 - (1 - F) ** best_ksi
A_best = best_ksi / best_sigma

plt.figure()
plt.plot(x, y_best, "+", label="Storm peaks")
plt.plot(x, A_best * x, "--", label="Fitted line")
plt.xlabel(r"$X = H_s-u$ [m]")
plt.ylabel(r"$Y$")
plt.title("Least-squares GPD fit")
plt.grid()
plt.legend()
plt.show()

########################################################################################
#--------------------------------- Cool ChatGPT diagnostics------------------------------
########################################################################################

# Plot the fitted GPD against empirical plotting positions

# Generate a smooth range of excess wave heights
x_curve = np.linspace(0, max(excesses) * 1.05, 500)

# Calculate the GPD cumulative distribution
F_curve = 1 - np.maximum(
    1 - best_ksi * x_curve / best_sigma, 0
) ** (1 / best_ksi)

# Calculate the fitted probabilities at observed points
F_fitted = 1 - np.maximum(
    1 - best_ksi * excesses / best_sigma, 0
) ** (1 / best_ksi)

# Plot
plt.figure(figsize=(9, 6))
plt.plot(excesses + u, plotting_position, "k+",
         label="Observed storm peaks", markersize=8)
plt.plot(x_curve + u, F_curve, "r-",
         label=fr"GPD fit ($\xi$={best_ksi:.4f}, $\sigma$={best_sigma:.4f} m)")
plt.xlabel(r"Significant wave height $H_s$ [m]")
plt.ylabel(r"Non-exceedance probability $F(H_s)$")
plt.title("Generalised Pareto Distribution: Fitted vs Observed")
plt.ylim(0, 1.02)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Additional goodness-of-fit diagnostics
print(f"Optimal ksi: {best_ksi:.5f}")
print(f"Optimal sigma: {best_sigma:.5f} m")
print(f"Average relative wave-height error: {smallest_error:.2%}")

########################################################################################
#--------------------------------- Cool ChatGPT diagnostics------------------------------
########################################################################################



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
