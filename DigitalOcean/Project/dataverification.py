
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from annomalities import (
	ANOMALY_YEAR,
	CELCIUS_OFFSET,
	DATA_DIR,
	FILE_CORE_NAME,
	MEASUREMENT_TIME_DIMS,
	YEARS,
	get_variable_name,
	load_data,
)

# ───────────────────────────────────────────────────────────────────────────────
# Data Verification Utility
# ───────────────────────────────────────────────────────────────────────────────
# This script validates SST (Sea Surface Temperature) data by comparing the
# distribution of values in the anomaly year against the baseline climatology
# (mean across reference years). Useful for:
# - Confirming data was loaded correctly
# - Visualizing how anomalous the target year is
# - Detecting data quality issues or outliers
# ───────────────────────────────────────────────────────────────────────────────


def _to_celsius_if_needed(values):
	"""Convert temperature from Kelvin to Celsius if values appear to be in Kelvin.
	
	Uses a simple heuristic: if the mean value is > 100, assumes the data is in Kelvin
	(since absolute temperature is always > 273) and subtracts the offset. Otherwise
	assuumes data is already in Celsius and returns unchanged.
	
	Args:
		values (ndarray): Temperature values to potentially convert
	
	Returns:
		ndarray: Temperature values in Celsius
	"""
	mean_val = float(np.nanmean(values))
	if mean_val > 100.0:
		return values - CELCIUS_OFFSET
	return values


def _year_mean_field_sst(filename_core, year):
	"""Load SST data and compute the 3-month mean spatial field for one year.
	
	Loads the NetCDF file for SST in the specified year, averages across all time
	dimensions (e.g., all days in the 3-month window), and converts to Celsius if needed.
	
	Args:
		filename_core (str): Base filename without extension or year
		year (int): Year to load
	
	Returns:
		ndarray: Mean SST field for the year [latitude, longitude] in degrees Celsius
	"""
	ds = load_data(filename_core, "sst", year)
	var_name = get_variable_name("sst")
	da = ds[var_name]

	time_dims = [d for d in MEASUREMENT_TIME_DIMS["sst"] if d in da.dims]
	year_mean = da.mean(dim=time_dims, skipna=True).values
	year_mean = _to_celsius_if_needed(year_mean)
	ds.close()
	return year_mean


def _flatten_valid(arr):
	"""Flatten a multi-dimensional array and remove NaN values.
	
	Useful for comparing distributions: converts a 2D spatial field [lat, lon]
	into a 1D array of valid measurements for statistical analysis.
	
	Args:
		arr (ndarray): Multi-dimensional array (typically [latitude, longitude])
	
	Returns:
		ndarray: 1D array of finite (non-NaN) values
	"""
	flat = np.asarray(arr).ravel()
	return flat[np.isfinite(flat)]


def plot_sst_distribution(base_values, anomaly_values, baseline_years, anomaly_year, out_path):
	"""Create side-by-side histogram and boxplot comparing baseline vs anomaly SST distributions.
	
	Visualizes the statistical properties of SST in the anomaly year versus the baseline
	climatology. Histograms show the probability density, with vertical lines marking
	each distribution's mean. Boxplots provide a compact summary of quartiles and spread.
	
	Args:
		base_values (ndarray): 1D array of SST values from baseline years
		anomaly_values (ndarray): 1D array of SST values from anomaly year
		baseline_years (list): List of years used for baseline
		anomaly_year (int): Year being analyzed for anomalies
		out_path (str or Path): Path to save the figure
	"""
	all_vals = np.concatenate([base_values, anomaly_values])
	bins = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 45)

	fig, axes = plt.subplots(1, 2, figsize=(13, 5))

	axes[0].hist(
		base_values,
		bins=bins,
		density=True,
		alpha=0.65,
		color="tab:blue",
		label=f"Baseline ({baseline_years[0]}-{baseline_years[-1]}, excl. {anomaly_year})",
	)
	axes[0].hist(
		anomaly_values,
		bins=bins,
		density=True,
		alpha=0.65,
		color="tab:red",
		label=f"Anomaly year ({anomaly_year})",
	)
	axes[0].axvline(np.nanmean(base_values), color="tab:blue", linestyle="--", linewidth=2)
	axes[0].axvline(np.nanmean(anomaly_values), color="tab:red", linestyle="--", linewidth=2)
	axes[0].set_title("SST Distribution of Mean Fields")
	axes[0].set_xlabel("SST (degC)")
	axes[0].set_ylabel("Density")
	axes[0].legend(frameon=False)

	axes[1].boxplot(
		[base_values, anomaly_values],
		tick_labels=["Baseline", f"{anomaly_year}"],
		showfliers=False,
	)
	axes[1].set_title("SST Distribution Comparison")
	axes[1].set_ylabel("SST (degC)")

	plt.tight_layout()
	plt.savefig(out_path, dpi=160)
	plt.show()


# ─── Configuration ──────────────────────────────────────────────────────────
# Output directory for verification results and plots
RESULTS_DIR = "DataResults"
def main():
	"""Main verification workflow for SST data.
	
	Orchestrates the complete data verification process:
	1. Load configuration (years, file paths, anomaly year)
	2. Compute baseline climatology (mean across reference years)
	3. Compute anomaly year mean field
	4. Extract and compare statistical distributions
	5. Generate comparison plots (histogram + boxplot)
	6. Print summary statistics
	
	This helps confirm that:
	- Data was loaded correctly
	- The anomaly year shows expected differences from baseline
	- There are no obvious data quality issues
	"""
	logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    
    
	project_dir = Path(__file__).resolve().parent
	filename_core = str(project_dir / DATA_DIR / FILE_CORE_NAME)
	out_dir = project_dir / RESULTS_DIR
	out_dir.mkdir(parents=True, exist_ok=True)

	print("Using configuration:")
	print(f"FILE_CORE_NAME={FILE_CORE_NAME}")
	print(f"ANOMALY_YEAR={ANOMALY_YEAR}")
	print(f"YEARS={YEARS}")
	print(f"Data prefix={filename_core}")
	print(f"Results dir={out_dir}")

	baseline_years = [y for y in YEARS if y != ANOMALY_YEAR]

	baseline_fields = [_year_mean_field_sst(filename_core, y) for y in baseline_years]
	baseline_climatology = np.nanmean(np.stack(baseline_fields, axis=0), axis=0)
	anomaly_field = _year_mean_field_sst(filename_core, ANOMALY_YEAR)

	baseline_values = _flatten_valid(baseline_climatology)
	anomaly_values = _flatten_valid(anomaly_field)

	print("SST distribution summary (degC)")
	print(f"Baseline years: {baseline_years}")
	print(f"Anomaly year: {ANOMALY_YEAR}")
	print(f"Baseline mean: {np.nanmean(baseline_values):.3f}")
	print(f"Anomaly-year mean: {np.nanmean(anomaly_values):.3f}")
	print(f"Mean difference (anomaly - baseline): {np.nanmean(anomaly_values) - np.nanmean(baseline_values):.3f}")

	out_path = out_dir / f"{FILE_CORE_NAME}_sst_distribution_baseline_vs_{ANOMALY_YEAR}.png"
	plot_sst_distribution(
		baseline_values,
		anomaly_values,
		baseline_years,
		ANOMALY_YEAR,
		out_path,
	)
	print(f"Saved figure: {out_path}")



if __name__ == "__main__":
	"""Execute the data verification workflow.
	
	This script is useful for:
	- Validating that data was loaded correctly
	- Understanding how different the anomaly year is from climatology
	- Identifying potential data quality issues before statistical analysis
	
	Output includes:
	- Summary statistics printed to console (mean SST values, anomaly magnitude)
	- Comparison plots saved to DataResults/ directory
	"""
	main()

