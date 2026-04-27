import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging


# ── User settings ───────────────────────────────────────────────────────────
# These settings control which data to analyze and how to aggregate it

AGG_FUNC = 'mean'  # Choose 'mean' or 'std' to aggregate daily samples into single maps

# The target year to compare against the baseline (reference years)
ANOMALY_YEAR = 2016

# File naming prefix - matches the data file naming convention
FILE_CORE_NAME = "mhw2016JFM"

# All years in the dataset. ANOMALY_YEAR will be compared against the other years.
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

# ANOMALY_YEAR = 2023
# FILE_CORE_NAME = "mhw2023JJA"
# YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ── Configuration ───────────────────────────────────────────────────────────
# Data source and output settings, plus measurement-specific metadata

# Where the input NetCDF files are located (relative path)
DATA_DIR = "Data"

# Where to save the output plots
RESULTS_DIR = "Results"

# List of variables to analyze (wind, sea surface temperature, total cloud cover)
AVALIABLE_MEASUREMENTS = ['wind_intensity', 'sst', 'tcc'] 

# NetCDF variable names for each measurement type
# Wind intensity is computed from u100 and v100, so it's not listed here
MEASUREMENT_VARIABLES = { 'sst': 'analysed_sst',
                          'tcc': 'tcc'}

# Dimension names in each NetCDF file that we stack together to get samples
# SST files have only 'time'; wind and cloud files have both 'time' and 'valid_time'
MEASUREMENT_TIME_DIMS = { 'sst': ['time'],
                          'tcc': ['time', 'valid_time'],
                          'wind_intensity': ['time', 'valid_time']}

# Units for display on plot labels
MEASUREMENT_UNITS = { 'sst': '°C',
                      'tcc': '(0–1)',
                      'wind_intensity': 'm/s'}

# Human-readable labels for each variable
MEASUREMENT_LABELS = { 'sst': 'Sea Surface Temperature',
                       'tcc': 'Total Cloud Cover',
                       'wind_intensity': 'Wind Intensity'}

CELCIUS_OFFSET = 273.15  # Convert Kelvin to Celsius by subtracting this value

CMAP_MEASURE = 'viridis'  # Colormap for mean/standard deviation plots

CMAP_STD = 'magma'  # Colormap for standard deviation (when AGG_FUNC = 'std')

CMAP_ANOMALY = 'RdBu_r'  # Colormap for anomalies (red-blue, diverging)

AGG_LABELS = { 'mean': 'Daily Mean', 'std': 'Daily Std Dev' }  # Labels for plot titles

# ── Data operations ───────────────────────────────────────────────────────────
# Functions to load and compute derived quantities from the raw data

def load_data(filename, measurement, year):
    """
    Load a single NetCDF file for a given measurement and year.
    Returns an xarray Dataset with all variables and dimensions from the file.
    """
    logging.info(f"Loading data for year {year} from file {filename}_{year}.nc")
    ds = xr.open_dataset(f"{filename}_{measurement}_{year}.nc")
    return ds


def get_variable_name(measurement):
    """
    Return the NetCDF variable name for a given measurement type.
    Raises an error if the measurement is unknown.
    """
    var_name = MEASUREMENT_VARIABLES.get(measurement)
    if var_name is None:
        raise ValueError(f"Unknown measurement '{measurement}'. Valid options: {list(MEASUREMENT_VARIABLES.keys())}")
    return var_name


def _load_wind_intensity(filename, year, agg='mean'):
    """
    Load u-wind (u100) and v-wind (v100) components, then compute wind speed magnitude.
    Wind speed = sqrt(u^2 + v^2)
    agg controls aggregation: 'mean' computes time-averaged speed, 'std' computes time-std
    Returns: (longitude, latitude, wind_speed_array)
    """
    ds_u = xr.open_dataset(f"{filename}_uwind_{year}.nc")
    ds_v = xr.open_dataset(f"{filename}_vwind_{year}.nc")
    time_dims = MEASUREMENT_TIME_DIMS['wind_intensity']
    agg_fn = lambda v: v.mean(dim=time_dims).values if agg == 'mean' else v.std(dim=time_dims).values
    u = agg_fn(ds_u['u100'])
    v = agg_fn(ds_v['v100'])
    lon = ds_u['longitude'].values
    lat = ds_u['latitude'].values
    ds_u.close()
    ds_v.close()
    return lon, lat, np.sqrt(u**2 + v**2)


def compute_avg(filename, measurement, year):
    """
    Compute the daily mean for a single measurement and year.
    For wind, this averages the magnitude after computation.
    For SST, converts from Kelvin to Celsius.
    Returns: (longitude, latitude, 2D mean array [lat, lon])
    """
    if measurement == 'wind_intensity':
        return _load_wind_intensity(filename, year, agg='mean')
    ds = load_data(filename, measurement, year)
    var_name = get_variable_name(measurement)
    var_data = ds[var_name]
    time_dims = MEASUREMENT_TIME_DIMS[measurement]
    avg_var = var_data.mean(dim=time_dims).values
    if measurement == 'sst':
        avg_var -= CELCIUS_OFFSET
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    ds.close()
    return lon, lat, avg_var

def compute_std(filename, measurement, year):
    """
    Compute the daily standard deviation for a single measurement and year.
    Similar to compute_avg but returns time-std instead of time-mean.
    Returns: (longitude, latitude, 2D std array [lat, lon])
    """
    if measurement == 'wind_intensity':
        return _load_wind_intensity(filename, year, agg='std')
    ds = load_data(filename, measurement, year)
    var_name = get_variable_name(measurement)
    var_data = ds[var_name]
    time_dims = MEASUREMENT_TIME_DIMS[measurement]
    std_var = var_data.std(dim=time_dims).values
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    ds.close()
    return lon, lat, std_var
   

def _compute_year(filename, measurement, year, agg_func):
    """
    Helper function that calls either compute_avg or compute_std based on AGG_FUNC.
    This centralizes the decision logic for aggregation method.
    """
    if agg_func == 'mean':
        return compute_avg(filename, measurement, year)
    return compute_std(filename, measurement, year)


def compute_baseline(filename, measurement, years):
    """
    Compute the baseline (reference) map by averaging across all input years.
    This is typically used to compare a single anomaly year against a multi-year baseline.
    Returns: (longitude, latitude, 2D baseline array [lat, lon])
    """
    all_yearly = []
    lon = lat = None
    for year in years:
        lo, la, val = _compute_year(filename, measurement, year, AGG_FUNC)
        all_yearly.append(val)
        if lon is None:
            lon, lat = lo, la
    stacked = np.stack(all_yearly, axis=0)
    baseline = np.nanmean(stacked, axis=0)
    return lon, lat, baseline


def compute_anomaly(filename, measurement, year, baseline):
    """
    Compute the anomaly for a single year relative to a baseline map.
    Anomaly = year_mean - baseline
    Positive values indicate the year is warmer/higher than the long-term average.
    Returns: 2D anomaly array [lat, lon]
    """
    _, _, year_avg = compute_avg(filename, measurement, year)
    return year_avg - baseline

# ── Plotting ──────────────────────────────────────────────────────────────────
# Functions to create and save publication-quality visualizations

def plot_heatmap(lon, lat, data, measurement, title, save_path, label=None):
    """
    Create a single-panel heatmap (pcolormesh) for a 2D data field.
    Automatically applies the appropriate colormap based on aggregation method.
    Saves the figure and displays it on screen.
    """
    if label is None:
        label = f'{AGG_LABELS[AGG_FUNC]} {MEASUREMENT_LABELS[measurement]} ({MEASUREMENT_UNITS[measurement]})'
    cmap = CMAP_STD if AGG_FUNC == 'std' else CMAP_MEASURE
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto')
    fig.colorbar(img, ax=ax, label=label)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_anomaly_heatmap(lon, lat, anomaly, measurement, title, save_path):
    """
    Create a single-panel heatmap for anomaly data using a diverging colormap.
    Anomalies are centered at zero, with symmetric color limits.
    """
    label = f'{MEASUREMENT_LABELS[measurement]} Anomaly ({MEASUREMENT_UNITS[measurement]})'
    vmax = np.nanmax(np.abs(anomaly))
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.pcolormesh(lon, lat, anomaly, cmap=CMAP_ANOMALY, shading='auto',
                        vmin=-vmax, vmax=vmax)
    fig.colorbar(img, ax=ax, label=label)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_year_vs_anomaly(lon, lat, year_avg, anomaly, measurement, year, save_path):
    """
    Create a side-by-side comparison figure showing:
    - LEFT: the target year's mean values
    - RIGHT: the anomaly (difference from baseline) for that year
    This makes it easy to see both the absolute values and their deviations.
    """
    units = MEASUREMENT_UNITS[measurement]
    mlabel = MEASUREMENT_LABELS[measurement]
    vmax_anom = np.nanmax(np.abs(anomaly))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

    cmap = CMAP_STD if AGG_FUNC == 'std' else CMAP_MEASURE
    img1 = ax1.pcolormesh(lon, lat, year_avg, cmap=cmap, shading='auto')
    fig.colorbar(img1, ax=ax1, label=f'{mlabel} ({units})')
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title(f'{AGG_LABELS[AGG_FUNC]} {mlabel} — JFM {year}\n(NW Atlantic, {YEARS[0]}–{YEARS[-1]} dataset)')

    img2 = ax2.pcolormesh(lon, lat, anomaly, cmap=CMAP_ANOMALY, shading='auto',
                          vmin=-vmax_anom, vmax=vmax_anom)
    fig.colorbar(img2, ax=ax2, label=f'{AGG_LABELS[AGG_FUNC]} Anomaly ({units})')
    ax2.set_xlabel('Longitude (°E)')
    ax2.set_ylabel('Latitude (°N)')
    ax2.set_title(f'{mlabel} Anomaly: {year} vs Baseline ({YEARS[0]}–{YEARS[-1]})\n{AGG_LABELS[AGG_FUNC]}, positive = above baseline')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main(fname, m, label):
    """
    Main workflow for a single measurement type:
    1. Compute baseline (average across all reference years)
    2. Compute target year average
    3. Compute anomaly (target - baseline)
    4. Create and save visualizations
    """
    agg = AGG_FUNC
    # Baseline across all years
    lon, lat, baseline = compute_baseline(fname, m, YEARS)
    plot_heatmap(lon, lat, baseline, m,
                 f'Baseline {AGG_LABELS[agg]} {label}\nJFM {YEARS[0]}–{YEARS[-1]} (NW Atlantic)',
                 f'{RESULTS_DIR}/{FILE_CORE_NAME}_{agg}_{m}_baseline.png')

    # Anomaly for ANOMALY_YEAR
    _, _, year_val = _compute_year(fname, m, ANOMALY_YEAR, agg)
    anomaly = year_val - baseline
    plot_year_vs_anomaly(lon, lat, year_val, anomaly, m, ANOMALY_YEAR,
                         f'{RESULTS_DIR}/{FILE_CORE_NAME}_{agg}_{m}_anomaly_{ANOMALY_YEAR}.png')
                        

if __name__ == "__main__":
    """
    Main execution block: loops through all measurements and creates analysis plots.
    """
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    fname = f"{DATA_DIR}/{FILE_CORE_NAME}"
    for m in AVALIABLE_MEASUREMENTS:
        label = MEASUREMENT_LABELS[m]
        main(fname, m, label)