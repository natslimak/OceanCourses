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
    return calculated_p - p

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
u = 1.255

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


# Define the excesses for the least-squares fit
x = excesses

ksi_range = np.linspace(0.0, 0.5,2000)

smallest_error = np.inf
best_ksi = None
best_sigma = None
errors = []
for ksi in ksi_range:
    y = 1 - (1 - plotting_position) ** ksi
    #A = np.sum(x * y) / np.sum(x ** 2)\
    A = np.polyfit(x, y, 1)[0]
    y_fit = x * A

    sigma = ksi / A

    # plt.plot(x, y, "+", label="Storm peaks")
    # plt.plot(x, y_fit, "--", label="Fitted line")
    # #plt.plot(x, slope * x + intercept, "r-", label="Fitted line")
    # plt.title("Fitting GPD to transformed data")
    # plt.xlabel(r"$X$")
    # plt.ylabel(r"$Y$")
    # plt.grid()
    # plt.show()

    # print(A)
    # print(f"ksi: {ksi:.4f}, sigma: {sigma:.4f}")

    pareto_real = 1 - (1 - (ksi * x / sigma)) ** (1 / ksi)

    
    error = (1 / n) * np.sum((np.abs(plotting_position - pareto_real)) / plotting_position)
    errors.append(error)
    if error < smallest_error:
        smallest_error = error
        best_ksi = ksi
        best_sigma = sigma
        # print(best_ksi, best_sigma)


print("Resuts of the parameter search:")
print(f"Smallest error: {smallest_error}")
print(f"ksi: {best_ksi}")
print(f"sigma: {best_sigma}")

np.polyfit(x,y, 1)
# Fit a line to x and y 
fit = np.polyfit(x, y, 1)
slope, intercept = fit
#slope = selected_ksi / selected_sigma

y_best = 1 - (1 - plotting_position) ** best_ksi
A_best = best_ksi / best_sigma
print(A_best)
y_best_fit = x * A_best
plt.plot(x, y_best, "+", label="Storm peaks")
plt.plot(x, y_best_fit, "--", label="Fitted line")
# Add ksi and sigma values to the plot
plt.text(0.1, 0.9, f"Best ksi: {best_ksi:.4f}\nBest sigma: {best_sigma:.4f}\nCalc A: {A_best:.4f}", transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
#plt.plot(x, slope * x + intercept, "r-", label="Fitted line")
plt.title("Fitting GPD to transformed data")
plt.xlabel(r"$X$")
plt.ylabel(r"$Y$")
plt.grid()
plt.show()


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


# =================================================================
# 3. Relationship between the Hs and Tp
# =================================================================

valid_wave = (
    np.isfinite(df_wave["Hs"])
    & np.isfinite(df_wave["Tp"])
    & (df_wave["Hs"] > 0)
    & (df_wave["Tp"] > 0)
)
fit_wave = df_wave.loc[valid_wave & (df_wave["Hs"] >= 0.2)]

def hs_tp_relationship(hs, alpha, beta):
    return alpha * hs ** beta


relationship_parameters, _ = curve_fit(
    hs_tp_relationship,
    fit_wave["Hs"],
    fit_wave["Tp"],
    p0=(3.4, 0.5),
    bounds=([0, 0], [np.inf, np.inf]),
    maxfev=10000,
)
alpha, beta = relationship_parameters

hs_fit = np.linspace(
    0,
    3,
    200,
)
tp_fit = alpha * hs_fit ** beta

plt.figure(figsize=(10, 6))
plt.scatter(
    df_wave.loc[valid_wave & (df_wave["Hs"] < 0.2), "Hs"],
    df_wave.loc[valid_wave & (df_wave["Hs"] < 0.2), "Tp"],
    color="lightskyblue",
    s=10,
    alpha=0.5,
    label=r"Excluded points ($H_s < 0.3$ m)",
)
plt.scatter(
    fit_wave["Hs"],
    fit_wave["Tp"],
    color="b",
    s=10,
    alpha=0.35,
    label=r"Fitting data ($H_s \geq 0.3$ m)",
)
plt.plot(
    hs_fit,
    tp_fit,
    color="r",
    label=fr"Fitted relationship: $T_p = {alpha:.2f}H_s^{{{beta:.2f}}}$",
)
plt.title("Relationship between Significant Wave Height (Hs) and Peak Wave Period (Tp)")
plt.xlabel("Significant Wave Height (Hs) [m]")
plt.ylabel("Peak Wave Period (Tp) [s]")
plt.xlim(0, 3)
plt.ylim(bottom=0)
plt.grid()
plt.legend()
plt.show()

print(f"Fitted relationship from {len(fit_wave)} observations (Hs >= 0.3 m): "
      f"Tp = {alpha:.4f} * Hs^{beta:.4f}")







# =================================================================
# 4. Estimate the design Hs and Tp
# =================================================================
x_TR = best_sigma/best_ksi * (1 - (peaks_per_year * T_R)** (-best_ksi)) + u
print(f"Design Hs for T = {T_R:.2f} years: {x_TR:.3f} m")