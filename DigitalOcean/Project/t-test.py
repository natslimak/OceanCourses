import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import ttest_ind


# ── User settings ───────────────────────────────────────────────────────────
# Configure the anomaly detection analysis by specifying which year is the target
# for anomaly detection and which years serve as the reference baseline.

# ANOMALY_YEAR: The year you want to test for anomalies (e.g., 2016 for marine heatwave)
# FILE_CORE_NAME: Prefix for NetCDF filenames (e.g., "mhw2016JFM" → "mhw2016JFM_sst_2016.nc")
# YEARS: Complete list of years available in your dataset. Reference years are derived
#        by excluding ANOMALY_YEAR from this list.

# Example configurations:
# ANOMALY_YEAR = 2016
# FILE_CORE_NAME = "mhw2016JFM"
# YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

ANOMALY_YEAR = 2023
FILE_CORE_NAME = "mhw2023JJA"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ALPHA: Statistical significance threshold (p-value cutoff). Typically 0.05.
# EQUAL_VAR: Set to False for Welch's t-test (doesn't assume equal variances).
#            Set to True for Student's t-test (assumes equal variances).
ALPHA = 0.05
EQUAL_VAR = False  # False = Welch t-test (recommended for unequal variances)

# ── Configuration ───────────────────────────────────────────────────────────
# File paths and measurement metadata. Do not modify unless your data structure
# or variable names differ from the standard oceanographic format.

DATA_DIR = "Data"  # Directory containing NetCDF files
RESULTS_DIR = "Results_ttest"  # Output directory for plots and results
AVAILABLE_MEASUREMENTS = ["wind_intensity", "sst", "tcc"]  # All measurements to analyze

# MEASUREMENT_VARIABLES: Maps measurement names to their variable names in NetCDF files.
# For example, "sst" data is stored under the "analysed_sst" variable in the file.
MEASUREMENT_VARIABLES = {
    "sst": "analysed_sst",
    "tcc": "tcc",
}

# MEASUREMENT_TIME_DIMS: Which dimensions in the NetCDF file represent time samples.
# sst has one time dimension ("time"), while tcc and wind use two ("time" and "valid_time").
# These are stacked to create a single "sample" dimension for statistical testing.
MEASUREMENT_TIME_DIMS = {
    "sst": ["time"],
    "tcc": ["time", "valid_time"],
    "wind_intensity": ["time", "valid_time"],
}

# MEASUREMENT_UNITS: Display units for each measurement in plot labels.
MEASUREMENT_UNITS = {
    "sst": "degC",
    "tcc": "(0-1)",
    "wind_intensity": "m/s",
}

# MEASUREMENT_LABELS: Human-readable names for each measurement in plot titles.
MEASUREMENT_LABELS = {
    "sst": "Sea Surface Temperature",
    "tcc": "Total Cloud Cover",
    "wind_intensity": "Wind Intensity",
}

# CELSIUS_OFFSET: Converts temperature from Kelvin to Celsius. Used only for SST.
CELSIUS_OFFSET = 273.15

# CMAP_MEASURE: Colormap for absolute measurement values (baseline, target means)
CMAP_MEASURE = "viridis"  # Good for positive values, intuitive progression
# CMAP_ANOMALY: Colormap for anomalies (difference from baseline)
CMAP_ANOMALY = "RdBu_r"  # Diverging: red=positive anomaly, blue=negative anomaly
# CMAP_PVALUE: Colormap for statistical p-values
CMAP_PVALUE = "plasma_r"  # Inverted plasma: low p-values are bright (significant)


# ── Data operations ─────────────────────────────────────────────────────────

def load_data(base_filename, measurement, year):
    """Load a NetCDF file for a specific measurement and year.
    
    Args:
        base_filename (str): Base filename without extension (e.g., "Data/mhw2023JJA")
        measurement (str): Measurement name ("sst", "tcc", or measurement in file)
        year (int): Year of data to load
    
    Returns:
        xarray.Dataset: Loaded NetCDF dataset
    """
    logging.info("Loading %s for year %s", measurement, year)
    return xr.open_dataset(f"{base_filename}_{measurement}_{year}.nc")


def get_variable_name(measurement):
    """Get the NetCDF variable name for a given measurement.
    
    For example, "sst" measurement uses the "analysed_sst" variable in the file.
    This function maps logical measurement names to their actual NetCDF variable names.
    
    Args:
        measurement (str): Measurement name ("sst", "tcc", etc.)
    
    Returns:
        str: NetCDF variable name
    
    Raises:
        ValueError: If measurement not found in MEASUREMENT_VARIABLES
    """
    var_name = MEASUREMENT_VARIABLES.get(measurement)
    if var_name is None:
        raise ValueError(
            f"Unknown measurement '{measurement}'. Valid options: {list(MEASUREMENT_VARIABLES.keys())}"
        )
    return var_name


def _stack_samples(da, time_dims, measurement):
    """Stack multi-dimensional time data into a single sample dimension.
    
    Converts data from [time, latitude, longitude] or [time, valid_time, latitude, longitude]
    into [samples, latitude, longitude] for statistical testing.
    
    Args:
        da (xarray.DataArray): Data array with multiple time dimensions
        time_dims (list): Time dimension names to stack (e.g., ["time"] or ["time", "valid_time"])
        measurement (str): Measurement name (used to determine if unit conversion needed)
    
    Returns:
        numpy.ndarray: Stacked samples array of shape [n_samples, n_lat, n_lon]
    """
    used_time_dims = [dim for dim in time_dims if dim in da.dims]
    if not used_time_dims:
        raise ValueError(f"No expected time dims found for {measurement}. Found dims: {da.dims}")

    stacked = da.stack(sample=used_time_dims).transpose("sample", "latitude", "longitude")
    values = stacked.values
    # Convert SST from Kelvin to Celsius
    if measurement == "sst":
        values = values - CELSIUS_OFFSET
    return values


def _load_wind_intensity_samples(base_filename, year):
    """Load wind components (u, v) and compute wind speed magnitude.
    
    Wind intensity is derived from zonal (u100) and meridional (v100) components
    as: speed = sqrt(u² + v²)
    
    Args:
        base_filename (str): Base filename without extension
        year (int): Year of wind data to load
    
    Returns:
        tuple: (lon, lat, wind_speed_samples) where wind_speed_samples shape is
               [n_samples, n_lat, n_lon]
    """
    ds_u = xr.open_dataset(f"{base_filename}_uwind_{year}.nc")
    ds_v = xr.open_dataset(f"{base_filename}_vwind_{year}.nc")

    try:
        lon = ds_u["longitude"].values
        lat = ds_u["latitude"].values
        time_dims = MEASUREMENT_TIME_DIMS["wind_intensity"]
        # Stack time dimensions and compute wind speed from components
        u = _stack_samples(ds_u["u100"], time_dims, "wind_intensity")
        v = _stack_samples(ds_v["v100"], time_dims, "wind_intensity")
        speed = np.sqrt(u ** 2 + v ** 2)  # Magnitude of wind vector
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
    """Load samples from reference years and target anomaly year.
    
    Loads data for all reference years (all YEARS except target_year) and concatenates
    them into a single array of reference samples. Also loads target year separately.
    These are used to compute baseline statistics and test for anomalies.
    
    Args:
        base_filename (str): Base filename without extension
        measurement (str): Measurement name ("sst", "tcc", "wind_intensity")
        target_year (int): Year to test for anomalies
        years (list): List of all available years
    
    Returns:
        tuple: (lon, lat, reference_samples, target_samples, reference_years)
               - lon: 1D array of longitudes
               - lat: 1D array of latitudes
               - reference_samples: 3D array [ref_samples, lat, lon]
               - target_samples: 3D array [target_samples, lat, lon]
               - reference_years: list of years used for baseline
    """
    reference_years = [y for y in years if y != target_year]
    if not reference_years:
        raise ValueError("Need at least one reference year different from ANOMALY_YEAR.")

    lon = lat = None
    reference_samples = []

    # Load and concatenate samples from all reference years
    for year in reference_years:
        lo, la, samples = load_samples_for_year(base_filename, measurement, year)
        if lon is None:
            lon, lat = lo, la
        reference_samples.append(samples)

    # Load target year separately for anomaly testing
    _, _, target_samples = load_samples_for_year(base_filename, measurement, target_year)
    reference_samples = np.concatenate(reference_samples, axis=0)
    return lon, lat, reference_samples, target_samples, reference_years


def compute_maps_and_ttest(reference_samples, target_samples, alpha=0.05, equal_var=False):
    """Compute means, anomalies, and perform grid-point Welch t-tests.
    
    Performs an independent samples t-test at each grid point comparing the
    target year distribution to the reference years distribution. This tests
    whether the target year is statistically significantly different from the
    baseline climatology at each location.
    
    Args:
        reference_samples (ndarray): Reference data [ref_samples, lat, lon]
        target_samples (ndarray): Target year data [target_samples, lat, lon]
        alpha (float): Significance threshold (default 0.05)
        equal_var (bool): If False, use Welch t-test (unequal variances).
                         If True, use Student t-test (equal variances assumed).
    
    Returns:
        tuple: (reference_mean, target_mean, anomaly, t_stat, p_values, significant_mask)
               - reference_mean: Mean of reference years [lat, lon]
               - target_mean: Mean of target year [lat, lon]
               - anomaly: target_mean - reference_mean [lat, lon]
               - t_stat: t-statistic at each grid point [lat, lon]
               - p_values: p-value at each grid point [lat, lon]
               - significant_mask: Boolean mask where p_values <= alpha [lat, lon]
    """
    reference_mean = np.nanmean(reference_samples, axis=0)
    target_mean = np.nanmean(target_samples, axis=0)
    # Anomaly is the difference between target year and baseline climatology
    anomaly = target_mean - reference_mean

    # Perform independent samples t-test at each grid point
    t_stat, p_values = ttest_ind(
        target_samples,
        reference_samples,
        axis=0,
        equal_var=equal_var,
        nan_policy="omit",
    )
    # Create a boolean mask for statistically significant locations (p <= alpha)
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
    """Overlay black dots on a map to mark statistically significant grid points.
    
    This helper function adds a stippling pattern (black dots) to an existing
    map/axis to indicate where the measurement shows statistically significant
    anomalies (where p <= ALPHA).
    
    Args:
        ax (matplotlib.axes.Axes): Axes object to draw on
        lon (ndarray): Longitude coordinates [n_lon]
        lat (ndarray): Latitude coordinates [n_lat]
        sig_mask (ndarray): Boolean mask of significant locations [n_lat, n_lon]
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    yy, xx = np.where(sig_mask)  # Find indices where mask is True
    if yy.size > 0:
        # Plot black dots at significant locations
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
    """Create a side-by-side comparison of baseline vs target year with significance markers.
    
    Left panel: Mean value across all reference years (baseline climatology)
    Right panel: Mean value for target year with black stippling marking statistically
                 significant differences from the baseline (p <= ALPHA)
    
    Args:
        lon, lat: Coordinate arrays
        reference_mean: Baseline mean [lat, lon]
        target_mean: Target year mean [lat, lon]
        significant_mask: Boolean mask of significant locations [lat, lon]
        measurement: Measurement name (for labels/units)
        target_year: Year being analyzed
        reference_years: List of baseline years
        save_path: Path to save the output figure
    """
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
    """Plot the anomaly field (target minus baseline) with significance stippling.
    
    Shows the magnitude and pattern of deviations from the baseline climatology.
    Red indicates positive anomalies (above baseline), blue indicates negative anomalies
    (below baseline). Black stippling marks where the anomaly is statistically significant.
    
    Args:
        lon, lat: Coordinate arrays
        anomaly: Difference field [target_mean - reference_mean] [lat, lon]
        significant_mask: Boolean mask of significant locations [lat, lon]
        measurement: Measurement name (for labels/units)
        target_year: Year being analyzed
        reference_years: List of baseline years
        save_path: Path to save the output figure
    """
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
    """Plot the p-value field from the t-test analysis at each grid point.
    
    Shows the statistical significance of differences at each location. Darker/brighter
    colors indicate more significant differences (lower p-values). The white dashed line
    marks the alpha threshold (default 0.05).
    
    Args:
        lon, lat: Coordinate arrays
        p_values: Statistical p-value at each grid point [lat, lon]
        measurement: Measurement name (for labels)
        target_year: Year being analyzed
        save_path: Path to save the output figure
    """
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
    """Plot all significant grid points from all measurements on a single map.
    
    Creates an overview map showing where each measurement (SST, wind intensity, and
    total cloud cover) exhibits statistically significant anomalies. Different markers
    distinguish each measurement:
    - Red circles: Sea Surface Temperature
    - Green squares: Wind Intensity
    - Blue triangles: Total Cloud Cover
    
    Args:
        results_by_measurement (dict): {measurement: (lon, lat, significant_mask)}
                                       significant_mask is [lat, lon] boolean array
        base_filename (str): Base filename (used for output path construction)
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
    """Main analysis pipeline for a single measurement.
    
    Orchestrates the complete workflow:
    1. Load reference and target year data
    2. Compute means, anomalies, and t-test statistics
    3. Generate three output plots:
       - Baseline vs target year comparison
       - Anomaly map with significance stippling
       - P-value distribution map
    4. Log summary statistics (% significant grid points)
    
    Args:
        base_filename (str): Base filename without extension
        measurement (str): Measurement name ("sst", "tcc", "wind_intensity")
    
    Returns:
        tuple: (lon, lat, significant_mask) for use in combined significance map
    """
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
    """Execute the complete t-test analysis workflow.
    
    Workflow:
    1. Configure logging to display warnings and errors
    2. Analyze each measurement (SST, wind intensity, total cloud cover) separately
    3. For each measurement:
       - Perform Welch t-test comparing target year vs reference years
       - Generate three diagnostic plots
       - Log the percentage of significant grid points
    4. Create a combined significance map overlaying all three measurements
    
    Output files are saved to Results_ttest/ directory.
    """
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    fname = f"{DATA_DIR}/{FILE_CORE_NAME}"
    
    # Store results from each measurement for combined visualization
    results_by_measurement = {}
    
    # Analyze each measurement independently
    for measurement_name in AVAILABLE_MEASUREMENTS:
        lon, lat, sig_mask = main(fname, measurement_name)
        results_by_measurement[measurement_name] = (lon, lat, sig_mask)
    
    # Create an overview map showing all significant regions across all measurements
    plot_combined_significance(results_by_measurement, fname)