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

def plot_heatmap(lon, lat, data, measurement, title, save_path, label=None, cmap=CMAP_MEASURE):
    if label is None:
        label = f"{MEASUREMENT_LABELS[measurement]} ({MEASUREMENT_UNITS[measurement]})"
    fig, ax = plt.subplots(figsize=(12, 7))
    img = ax.pcolormesh(lon, lat, data, cmap=cmap, shading="auto")
    fig.colorbar(img, ax=ax, label=label)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def _add_significance_stipple(ax, lon, lat, sig_mask):
    lon2d, lat2d = np.meshgrid(lon, lat)
    yy, xx = np.where(sig_mask)
    if yy.size > 0:
        ax.scatter(
            lon2d[yy, xx],
            lat2d[yy, xx],
            s=6,
            c="black",
            alpha=0.95,
            marker="o",
            linewidths=0,
            label=f"p <= {ALPHA}",
        )


def plot_baseline_vs_target_with_significance(
    lon,
    lat,
    reference_mean,
    target_mean,
    significant_mask,
    measurement,
    target_year,
    reference_years,
    save_path,
):
    units = MEASUREMENT_UNITS[measurement]
    mlabel = MEASUREMENT_LABELS[measurement]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

    img1 = ax1.pcolormesh(lon, lat, reference_mean, cmap=CMAP_MEASURE, shading="auto")
    fig.colorbar(img1, ax=ax1, label=f"{mlabel} ({units})")
    ax1.set_xlabel("Longitude (degE)")
    ax1.set_ylabel("Latitude (degN)")
    ax1.set_title(f"Baseline Mean {mlabel} ({reference_years[0]}-{reference_years[-1]})")

    img2 = ax2.pcolormesh(lon, lat, target_mean, cmap=CMAP_MEASURE, shading="auto")
    fig.colorbar(img2, ax=ax2, label=f"{mlabel} ({units})")
    _add_significance_stipple(ax2, lon, lat, significant_mask)
    ax2.set_xlabel("Longitude (degE)")
    ax2.set_ylabel("Latitude (degN)")
    ax2.set_title(
        f"{mlabel} Mean: {target_year} vs Baseline\n"
        f"Stippling: statistically significant (p <= {ALPHA})"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_anomaly_with_significance(
    lon,
    lat,
    anomaly,
    significant_mask,
    measurement,
    target_year,
    reference_years,
    save_path,
):
    units = MEASUREMENT_UNITS[measurement]
    mlabel = MEASUREMENT_LABELS[measurement]
    vmax_anom = np.nanmax(np.abs(anomaly))

    fig, ax = plt.subplots(figsize=(12, 7))
    img = ax.pcolormesh(
        lon,
        lat,
        anomaly,
        cmap=CMAP_ANOMALY,
        shading="auto",
        vmin=-vmax_anom,
        vmax=vmax_anom,
    )
    fig.colorbar(img, ax=ax, label=f"Anomaly ({units})")
    _add_significance_stipple(ax, lon, lat, significant_mask)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.set_title(
        f"{mlabel} Anomaly: {target_year} vs {reference_years[0]}-{reference_years[-1]}\n"
        f"Stippling: statistically significant (p <= {ALPHA})"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


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


def plot_combined_significance(results_by_measurement, base_filename):
    """
    Plot all significant grid points from all measurements on a single map.
    results_by_measurement: dict of {measurement: (lon, lat, significant_mask)}
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    marker_styles = {
        "sst": {"marker": "o", "color": "red", "zorder": 1},
        "wind_intensity": {"marker": "s", "color": "green", "zorder": 2},
        "tcc": {"marker": "^", "color": "blue", "zorder": 3},
    }

    draw_order = ["sst", "wind_intensity", "tcc"]
    for measurement in draw_order:
        if measurement not in results_by_measurement:
            continue
        lon, lat, sig_mask = results_by_measurement[measurement]
        lon2d, lat2d = np.meshgrid(lon, lat)
        yy, xx = np.where(sig_mask)
        if yy.size > 0:
            style = marker_styles[measurement]
            ax.scatter(
                lon2d[yy, xx],
                lat2d[yy, xx],
                s=28,
                c=style["color"],
                marker=style["marker"],
                alpha=0.5,
                edgecolors="black",
                linewidths=0.35,
                label=MEASUREMENT_LABELS[measurement],
                zorder=style["zorder"],
            )

    ax.set_xlabel("Longitude (degE)", fontsize=12)
    ax.set_ylabel("Latitude (degN)", fontsize=12)
    ax.set_title(
        f"Combined Significance Map: All Variables (p <= {ALPHA})\n"
        f"Target Year: {ANOMALY_YEAR}",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{RESULTS_DIR}/{FILE_CORE_NAME}_combined_significance_map_{ANOMALY_YEAR}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    logging.warning(f"Combined significance map saved to {save_path}")


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

    plot_baseline_vs_target_with_significance(
        lon,
        lat,
        reference_mean,
        target_mean,
        significant_mask,
        measurement,
        ANOMALY_YEAR,
        reference_years,
        f"{RESULTS_DIR}/{FILE_CORE_NAME}_{measurement}_baseline_vs_target_{ANOMALY_YEAR}.png",
    )

    plot_anomaly_with_significance(
        lon,
        lat,
        anomaly,
        significant_mask,
        measurement,
        ANOMALY_YEAR,
        reference_years,
        f"{RESULTS_DIR}/{FILE_CORE_NAME}_{measurement}_anomaly_significance_{ANOMALY_YEAR}.png",
    )

    plot_p_value_map(
        lon,
        lat,
        p_values,
        measurement,
        ANOMALY_YEAR,
        f"{RESULTS_DIR}/{FILE_CORE_NAME}_{measurement}_pvalues_{ANOMALY_YEAR}.png",
    )

    return (lon, lat, significant_mask)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    fname = f"{DATA_DIR}/{FILE_CORE_NAME}"
    
    # Store results for combined plot
    results_by_measurement = {}
    
    for measurement_name in AVAILABLE_MEASUREMENTS:
        lon, lat, sig_mask = main(fname, measurement_name)
        results_by_measurement[measurement_name] = (lon, lat, sig_mask)
    
    # Create combined significance map
    plot_combined_significance(results_by_measurement, fname)