import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# ── Data Preparation ───────────────────────────────────────────────────────────

# Load the data from the text file 
names = ['wavelength', 'mq_01', 'chemocean_06', 'chemocean_07', 'chemocean_08',
	    'chemocean_09', 'chemocean_10', 'chemocean_NO3-0', 'chemocean_NO3-1',
	    'chemocean_NO3-2', 'chemocean_NO3-3', 'chemocean_NO3-4',
	    'chemocean_Br-1', 'chemocean_Br-2', 'chemocean_Br-3', 'chemocean_Br-0']

# skip the first row (header/metadata) and assign our 16 column names
ds = pd.read_csv("data/Group2.txt", delim_whitespace=True, header=None, skiprows=1, names=names)

# Separate the data into nitrate, bromide, and sample datasets
ds_nitrate = ds.filter(regex='NO3')
ds_bromide = ds.filter(regex='Br-')
ds_samples = ds.filter(regex='chemocean_[0-9]+')

conc_nitrate = [0, 6.3, 12.6, 18.9, 31.5]
conc_bromide = [4.8, 9.7, 13.6, 0]
sample_salinity = [33.9, 33.9, 33.8, 29.0, 16.5]


# ── Plotting Functions ───────────────────────────────────────────────────────────
def plot_raw_spectra(ds):
    wavelengths = ds['wavelength']
    absorbance_columns = [col for col in ds.columns if col.startswith('chemocean')]
    for col in absorbance_columns:
        plt.plot(wavelengths, ds[col], label=col)
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Raw Spectra')
    plt.legend()
    plt.show()


def get_absorbance_coefficient(ds, concentration_values, absorbance_columns, compound_name="Compound"):
    target_wavelength = 220 # Choose a wavelength

    # Get the row for that wavelength (find nearest if exact match doesn't exist)
    row = ds.loc[(ds['wavelength'] - target_wavelength).abs().idxmin()]

    x = np.array(concentration_values)      # Load the concentration values as a numpy array
    y = row[list(absorbance_columns)].to_numpy()    # Load absorbance values for the specified columns
    m, b = np.polyfit(x, y, 1)   # fit line

    # Plot
    plt.scatter(x, y, label="data")
    plt.plot(x, m * x + b, label=f"fit: y={m:.4f}x+{b:.4f}")
    plt.xlabel("Concentration (µM)")
    plt.ylabel("Absorbance")
    plt.title(f"{compound_name} calibration at {target_wavelength} nm")
    plt.legend()
    plt.show()

    return m


def get_calibration_slopes(ds, concentration_values, absorbance_columns, compound_name="Compound"):
    """Return the calibration slope (epsilon * L) for every wavelength."""
    x = np.array(concentration_values)
    wavelengths = ds["wavelength"].to_numpy()
    slopes = []

    for wavelength in wavelengths:
        row = ds.loc[(ds["wavelength"] - wavelength).abs().idxmin()]
        y = row[list(absorbance_columns)].to_numpy()
        if len(x) != len(y):
            raise ValueError(
                f"{compound_name}: {len(x)} concentrations but {len(y)} absorbance values"
            )
        m, _ = np.polyfit(x, y, 1)
        slopes.append(m)

    return pd.DataFrame({"wavelength": wavelengths, f"{compound_name}_slope": slopes})


def remove_bromide(ds, bromide_slopes, sample_columns, sample_salinity, path_length=1.0):
    """Subtract bromide absorbance from each sample spectrum."""
    corrected = pd.DataFrame({"wavelength": ds["wavelength"]})
    epsilon_br = bromide_slopes.set_index("wavelength")["Bromide_slope"]

    for sample_col, salinity in zip(sample_columns, sample_salinity):
        c_br = salinity * 23.25
        a_br = ds["wavelength"].map(epsilon_br) * path_length * c_br
        corrected[f"{sample_col}_no3_dom"] = ds[sample_col] - a_br

    # Plot the corrected spectra
    wavelengths = corrected["wavelength"]
    for col in corrected.columns:
        if col != "wavelength":
            plt.plot(wavelengths, corrected[col], label=col)

    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Corrected Spectra')
    plt.legend()
    plt.show()

    return corrected


# ── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1: Plot the raw spectra to visually inspect the data
    plot_raw_spectra(ds)

    # STEP 2: Calculate the absorbance coefficients (slopes) for nitrate and bromide
    nitrate_slope = get_absorbance_coefficient(ds, conc_nitrate, ds_nitrate.columns, "Nitrate")
    bromide_slope = get_absorbance_coefficient(ds, conc_bromide, ds_bromide.columns, "Bromide")

    # STEP 3: Remove Bromide
    bromide_slopes = get_calibration_slopes(ds, conc_bromide, ds_bromide.columns, "Bromide")
    corrected_spectra = remove_bromide(ds, bromide_slopes, ds_samples.columns, sample_salinity)

    print(corrected_spectra.head())



