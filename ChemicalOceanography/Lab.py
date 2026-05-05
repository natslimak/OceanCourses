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


def get_absorbance_coefficient(ds, concentration_values, absorbance_columns, compound_name="Compound", plot_wavelength=220):
    """Fit A = mC + b at every wavelength.
    
    - Computes slope (epsilon*L) for all wavelengths
    - Shows calibration plot at plot_wavelength for validation
    - Returns array of slopes (same length as ds)
    """
    x = np.asarray(concentration_values)
    slopes = []

    for i in range(len(ds)):
        y = ds.iloc[i][list(absorbance_columns)].to_numpy()
        m, _ = np.polyfit(x, y, 1)
        slopes.append(m)

    slopes_array = np.asarray(slopes)

    # Show calibration at one wavelength for validation
    row = ds.loc[(ds['wavelength'] - plot_wavelength).abs().idxmin()]
    y_plot = row[list(absorbance_columns)].to_numpy()
    m_plot, b_plot = np.polyfit(x, y_plot, 1)

    plt.scatter(x, y_plot, label="data")
    plt.plot(x, m_plot * x + b_plot, label=f"fit: y={m_plot:.4f}x+{b_plot:.4f}")
    plt.xlabel("Concentration (µM)")
    plt.ylabel("Absorbance")
    plt.title(f"{compound_name} calibration at {plot_wavelength} nm")
    plt.legend()
    plt.show()

    return slopes_array



def remove_bromide(ds, bromide_slopes, sample_columns, sample_salinity, path_length=1.0):
    """Subtract bromide absorbance from each sample spectrum.
    
    bromide_slopes: array-like of slopes (epsilon*L) for each wavelength
    """
    corrected = pd.DataFrame({"wavelength": ds["wavelength"]})
    eps_br = np.asarray(bromide_slopes)

    for sample_col, salinity in zip(sample_columns, sample_salinity):
        c_br = salinity * 23.25
        a_br = eps_br * path_length * c_br
        corrected[f"{sample_col}_no3_dom"] = ds[sample_col].to_numpy() - a_br

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



def smooth(y, window=7):
    """Simple moving-average smoothing (returns same-length array)."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def plot_epsilon(wavelengths, epsilon, label=None, smooth_window=9):
    """Plot molar absorptivity ε(λ) with raw points and a smoothed curve."""
    wl = np.asarray(wavelengths)
    eps = np.asarray(epsilon)

    plt.figure()
    plt.scatter(wl, eps, s=18, color='C1', alpha=0.8, label=f'{label} points' if label else 'points')
    eps_s = smooth(eps, window=smooth_window)
    plt.plot(wl, eps_s, color='C0', lw=1.8, label=f'{label} (smoothed)' if label else 'smoothed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Molar absorptivity ε(λ) (slope / L)')
    plt.title('Molar absorptivity ε(λ)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1: Plot the raw spectra to visually inspect the data
    plot_raw_spectra(ds)

    # STEP 2: Calculate the absorbance coefficients (slopes) for the full spectrum
    nitrate_slopes = get_absorbance_coefficient(ds, conc_nitrate, ds_nitrate.columns, "Nitrate")
    bromide_slopes = get_absorbance_coefficient(ds, conc_bromide, ds_bromide.columns, "Bromide")

    # STEP 2b: Plot ε(λ) for nitrate and bromide
    plot_epsilon(ds['wavelength'], nitrate_slopes, label='Nitrate')
    plot_epsilon(ds['wavelength'], bromide_slopes, label='Bromide')

    # STEP 3: Remove Bromide using the full-spectrum bromide slopes
    corrected_spectra = remove_bromide(ds, bromide_slopes, ds_samples.columns, sample_salinity)

    slopes = []
    for i, wavelength in enumerate(ds['wavelength']):
        current_slope = nitrate_slopes[i]
        slopes.append(current_slope)

    plt.figure()
    plt.plot(ds['wavelength'], slopes, 'o', color='C1', alpha=0.8, markersize=8)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Molar absorptivity ε(λ) (slope / L)')
    plt.title('Molar absorptivity ε(λ)')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()