
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


def _to_celsius_if_needed(values):
	"""Convert Kelvin to Celsius using a simple threshold check."""
	mean_val = float(np.nanmean(values))
	if mean_val > 100.0:
		return values - CELCIUS_OFFSET
	return values


def _year_mean_field_sst(filename_core, year):
	"""Return the 3-month SST mean map for one year."""
	ds = load_data(filename_core, "sst", year)
	var_name = get_variable_name("sst")
	da = ds[var_name]

	time_dims = [d for d in MEASUREMENT_TIME_DIMS["sst"] if d in da.dims]
	year_mean = da.mean(dim=time_dims, skipna=True).values
	year_mean = _to_celsius_if_needed(year_mean)
	ds.close()
	return year_mean


def _flatten_valid(arr):
	"""Flatten to 1D and drop NaNs."""
	flat = np.asarray(arr).ravel()
	return flat[np.isfinite(flat)]


def plot_sst_distribution(base_values, anomaly_values, baseline_years, anomaly_year, out_path):
	"""Plot histogram and boxplot comparing baseline and anomaly SST distributions."""
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

RESULTS_DIR = "DataResults"
def main():
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
	main()

