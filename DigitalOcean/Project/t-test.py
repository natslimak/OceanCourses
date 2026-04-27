import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import ttest_ind


# ── User settings ───────────────────────────────────────────────────────────

ANOMALY_YEAR = 2016
FILE_CORE_NAME = "mhw2016JFM"
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

# ANOMALY_YEAR = 2023
# FILE_CORE_NAME = "mhw2023JJA"
# YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

ALPHA = 0.05
EQUAL_VAR = False  # False = Welch t-test

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "Data"
RESULTS_DIR = "Results_ttest"
AVAILABLE_MEASUREMENTS = ["wind_intensity", "sst", "tcc"]

# For each measurement, define the variable name in the NetCDF file.
MEASUREMENT_VARIABLES = {
    "sst": "analysed_sst",
    "tcc": "tcc",
}

# Time/sample dims by measurement.
MEASUREMENT_TIME_DIMS = {
    "sst": ["time"],
    "tcc": ["time", "valid_time"],
    "wind_intensity": ["time", "valid_time"],
}

MEASUREMENT_UNITS = {
    "sst": "degC",
    "tcc": "(0-1)",
    "wind_intensity": "m/s",
}

MEASUREMENT_LABELS = {
    "sst": "Sea Surface Temperature",
    "tcc": "Total Cloud Cover",
    "wind_intensity": "Wind Intensity",
}

CELSIUS_OFFSET = 273.15

CMAP_MEASURE = "viridis"
CMAP_ANOMALY = "RdBu_r"
CMAP_PVALUE = "plasma_r"


# ── Data operations ─────────────────────────────────────────────────────────

def load_data(base_filename, measurement, year):
    logging.info("Loading %s for year %s", measurement, year)
    return xr.open_dataset(f"{base_filename}_{measurement}_{year}.nc")


def get_variable_name(measurement):
    var_name = MEASUREMENT_VARIABLES.get(measurement)
    if var_name is None:
        raise ValueError(
            f"Unknown measurement '{measurement}'. Valid options: {list(MEASUREMENT_VARIABLES.keys())}"
        )
    return var_name


def _stack_samples(da, time_dims, measurement):
    used_time_dims = [dim for dim in time_dims if dim in da.dims]
    if not used_time_dims:
        raise ValueError(f"No expected time dims found for {measurement}. Found dims: {da.dims}")

    stacked = da.stack(sample=used_time_dims).transpose("sample", "latitude", "longitude")
    values = stacked.values
    if measurement == "sst":
        values = values - CELSIUS_OFFSET
    return values


def _load_wind_intensity_samples(base_filename, year):
    ds_u = xr.open_dataset(f"{base_filename}_uwind_{year}.nc")
    ds_v = xr.open_dataset(f"{base_filename}_vwind_{year}.nc")

    try:
        lon = ds_u["longitude"].values
        lat = ds_u["latitude"].values
        time_dims = MEASUREMENT_TIME_DIMS["wind_intensity"]
        u = _stack_samples(ds_u["u100"], time_dims, "wind_intensity")
        v = _stack_samples(ds_v["v100"], time_dims, "wind_intensity")
        speed = np.sqrt(u ** 2 + v ** 2)
    finally:
        ds_u.close()
        ds_v.close()

    return lon, lat, speed


def load_samples_for_year(base_filename, measurement, year):
    """
    Return (lon, lat, samples) where samples shape is [n_samples, n_lat, n_lon].
    """
    if measurement == "wind_intensity":
        return _load_wind_intensity_samples(base_filename, year)

    ds = load_data(base_filename, measurement, year)
    try:
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        var_name = get_variable_name(measurement)
        time_dims = MEASUREMENT_TIME_DIMS[measurement]
        samples = _stack_samples(ds[var_name], time_dims, measurement)
    finally:
        ds.close()

    return lon, lat, samples


def load_reference_and_target(base_filename, measurement, target_year, years):
    reference_years = [y for y in years if y != target_year]
    if not reference_years:
        raise ValueError("Need at least one reference year different from ANOMALY_YEAR.")

    lon = lat = None
    reference_samples = []

    for year in reference_years:
        lo, la, samples = load_samples_for_year(base_filename, measurement, year)
        if lon is None:
            lon, lat = lo, la
        reference_samples.append(samples)

    _, _, target_samples = load_samples_for_year(base_filename, measurement, target_year)
    reference_samples = np.concatenate(reference_samples, axis=0)
    return lon, lat, reference_samples, target_samples, reference_years


def compute_maps_and_ttest(reference_samples, target_samples, alpha=0.05, equal_var=False):
    reference_mean = np.nanmean(reference_samples, axis=0)
    target_mean = np.nanmean(target_samples, axis=0)
    anomaly = target_mean - reference_mean

    t_stat, p_values = ttest_ind(
        target_samples,
        reference_samples,
        axis=0,
        equal_var=equal_var,
        nan_policy="omit",
    )
    significant_mask = p_values <= alpha

    return reference_mean, target_mean, anomaly, t_stat, p_values, significant_mask


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_p_value_map(lon, lat, p_values, measurement, target_year, save_path):
    mlabel = MEASUREMENT_LABELS[measurement]
    fig, ax = plt.subplots(figsize=(12, 7))
    img = ax.pcolormesh(lon, lat, p_values, cmap=CMAP_PVALUE, shading="auto", vmin=0.0, vmax=0.1)
    cbar = fig.colorbar(img, ax=ax, label="p-value")
    cbar.ax.axhline(ALPHA, color="white", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.set_title(f"{mlabel} Welch t-test p-values ({target_year} vs reference years)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main(base_filename, measurement):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lon, lat, reference_samples, target_samples, reference_years = load_reference_and_target(
        base_filename,
        measurement,
        ANOMALY_YEAR,
        YEARS,
    )

    (
        reference_mean,
        target_mean,
        anomaly,
        _t_stat,
        p_values,
        significant_mask,
    ) = compute_maps_and_ttest(
        reference_samples,
        target_samples,
        alpha=ALPHA,
        equal_var=EQUAL_VAR,
    )

    mlabel = MEASUREMENT_LABELS[measurement]
    n_sig = int(np.count_nonzero(significant_mask))
    n_total = int(np.count_nonzero(np.isfinite(p_values)))
    sig_pct = 100.0 * n_sig / n_total if n_total else np.nan
    logging.warning(
        "%s: significant grid points = %s/%s (%.2f%%) at alpha=%s",
        measurement,
        n_sig,
        n_total,
        sig_pct,
        ALPHA,
    )

    plot_p_value_map(
        lon,
        lat,
        p_values,
        measurement,
        ANOMALY_YEAR,
        f"{RESULTS_DIR}/{FILE_CORE_NAME}_{measurement}_pvalues_{ANOMALY_YEAR}.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    fname = f"{DATA_DIR}/{FILE_CORE_NAME}"
    for measurement_name in AVAILABLE_MEASUREMENTS:
        main(fname, measurement_name)