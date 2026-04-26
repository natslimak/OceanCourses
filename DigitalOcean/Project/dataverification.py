
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging


# ── User settings ───────────────────────────────────────────────────────────

AGG_FUNC = 'mean'  # 'mean' or 'std'
ANOMALY_YEAR = 2023

FILE_CORE_NAME = "mhw2023JJA"

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "Data"
RESULTS_DIR = "Results"
AVALIABLE_MEASUREMENTS = ['wind_intensity', 'sst', 'tcc'] 

#for each measurements diferent var is used
MEASUREMENT_VARIABLES = { 'sst': 'analysed_sst',
                          'tcc': 'tcc'}

# SST uses 'time', others use 'time' + 'valid_time'
MEASUREMENT_TIME_DIMS = { 'sst': ['time'],
                          'tcc': ['time', 'valid_time'],
                          'wind_intensity': ['time', 'valid_time']}

MEASUREMENT_UNITS = { 'sst': '°C',
                      'tcc': '(0–1)',
                      'wind_intensity': 'm/s'}

MEASUREMENT_LABELS = { 'sst': 'Sea Surface Temperature',
                       'tcc': 'Total Cloud Cover',
                       'wind_intensity': 'Wind Intensity'}

CELCIUS_OFFSET = 273.15

CMAP_MEASURE= 'viridis'

CMAP_STD = 'magma'

CMAP_ANOMALY = 'RdBu_r'

AGG_LABELS = { 'mean': 'Daily Mean', 'std': 'Daily Std Dev' }

# ── Data operations ───────────────────────────────────────────────────────────

def load_data(filename, measurement, year):
    logging.info(f"Loading data for year {year} from file {filename}_{year}.nc")
    ds = xr.open_dataset(f"{filename}_{measurement}_{year}.nc")
    return ds


def get_variable_name(measurement):
    var_name = MEASUREMENT_VARIABLES.get(measurement)
    if var_name is None:
        raise ValueError(f"Unknown measurement '{measurement}'. Valid options: {list(MEASUREMENT_VARIABLES.keys())}")
    return var_name