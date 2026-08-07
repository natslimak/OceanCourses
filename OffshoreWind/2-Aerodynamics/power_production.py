'''import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend so importing user scripts doesn't open GUI windows
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import importlib.util

# Read constants/paths from ExcelFunctions if available
EXCEL_PATH = 'IEA-22-280-RWT_tabular.xlsx'
SHEET_NAME = 'Rotor Performance'

# Columns in the Excel sheet
WIND_COL = 0  # A
POWER_COL = 2 # C
POWERCOEFF_COL = 3 # D


def read_power_curve(excel_path=EXCEL_PATH, sheet_name=SHEET_NAME):
    """Read the power curve (wind speed vs power) from Excel.
    Returns wind_speeds, powers arrays.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    wind = df.iloc[:, WIND_COL].values.astype(float)
    power = df.iloc[:, POWER_COL].values.astype(float)
    power_coeff = df.iloc[:, POWERCOEFF_COL].values.astype(float)
    # clean NaNs
    mask = ~np.isnan(wind) & ~np.isnan(power)
    wind = wind[mask]
    power = power[mask]
    power_coeff = power_coeff[mask]
    # ensure sorted by wind
    sort_idx = np.argsort(wind)
    return wind[sort_idx], power[sort_idx], power_coeff[sort_idx]


def power_from_power_curve(ws, wind_curve=None, power_curve=None, excel_path=EXCEL_PATH, sheet_name=SHEET_NAME):
    """Interpolate power (absolute) for given wind speed(s) ws using the turbine power curve.
    ws: scalar or array of wind speeds
    If wind_curve/power_curve are not provided, they will be read from Excel.
    Returns interpolated power (same shape as ws).
    """
    if wind_curve is None or power_curve is None:
        wind_curve, power_curve = read_power_curve(excel_path, sheet_name)
    f = interp1d(wind_curve, power_curve, bounds_error=False, fill_value=(power_curve[0], power_curve[-1]))
    return f(ws)

'''
def rotor_area_average_from_les(les_dict, x_R, rotor_radius=1.0):
    """Compute rotor-area-averaged velocity from LES fields at downstream position x_R (in R).
    les_dict must contain 'x','y','z','meanU'.
    Returns average velocity (m/s).
    """
    x_arr = les_dict['x'][:,0,0]
    xi = (np.abs(x_arr - x_R)).argmin()
    Y = np.array(les_dict['y'][xi,:,:], copy=True)
    Z = np.array(les_dict['z'][xi,:,:], copy=True)
    Uplane = np.array(les_dict['meanU'][xi,:,:], copy=True)
    mask = (Y**2 + Z**2) <= rotor_radius**2
    if mask.sum() == 0:
        return np.nan
    return np.mean(Uplane[mask])


def rotor_area_average_from_model(deficit_field, les_dict, x_R, rotor_radius=1.0, freestream_velocity=None):
    """Compute rotor-area-averaged velocity for a given model deficit field at x_R.
    deficit_field: a 3D array matching les['x','y','z'] or a function that accepts (Y,Z) and returns deficit.
    les_dict: to extract Y,Z grid for the given x.
    freestream_velocity: optional scalar free-stream velocity (m/s). If None, will use les_dict['meanU'][0,0,0] as fallback.
    Returns average model velocity (m/s).
    """
    x_arr = les_dict['x'][:,0,0]
    xi = (np.abs(x_arr - x_R)).argmin()
    Y = np.array(les_dict['y'][xi,:,:], copy=True)
    Z = np.array(les_dict['z'][xi,:,:], copy=True)
    mask = (Y**2 + Z**2) <= rotor_radius**2
    if isinstance(deficit_field, np.ndarray):
        def_plane = np.array(deficit_field[xi,:,:], copy=True)
    else:
        def_plane = np.array(deficit_field(Y, Z), copy=True)
    if mask.sum() == 0:
        return np.nan
    def_vals = np.nan_to_num(def_plane[mask], nan=0.0)
    if freestream_velocity is None:
        freestream_velocity = float(les_dict['meanU'][0,0,0])
    return np.mean((1.0 - def_vals) * freestream_velocity)
'''

def estimate_power_with_powercurve(les_dict, noj_def, gauss_def, x_values, rotor_radius=1.0, use_powercurve=True):
    """Estimate absolute power (from power curve) and normalized power (U^3) for LES and models across x_values (in R).
    les_dict: dictionary with LES arrays
    noj_def, gauss_def: 3D arrays of deficits matching les grid
    x_values: iterable of downstream x (in R)
    Returns dict with keys 'x','vel_les','vel_noj','vel_gauss','P_les','P_noj','P_gauss' (absolute power if powercurve available, else normalized cubic power)
    """
    vel_les = []
    vel_noj = []
    vel_gauss = []
    for x_R in x_values:
        x_arr = les_dict['x'][:,0,0]
        xi = (np.abs(x_arr - x_R)).argmin()
        Y = np.array(les_dict['y'][xi,:,:], copy=True)
        Z = np.array(les_dict['z'][xi,:,:], copy=True)
        mask = (Y**2 + Z**2) <= rotor_radius**2
        # LES
        vel_les.append(np.mean(np.array(les_dict['meanU'][xi,:,:], copy=True)[mask]))
        # NOJ and GA - make defensive copies in case inputs are read-only
        noj_plane = np.array(noj_def[xi,:,:], copy=True)
        ga_plane = np.array(gauss_def[xi,:,:], copy=True)
        vel_noj.append(np.mean(float(les_dict['meanU'][0,0,0]) * (1 - np.nan_to_num(noj_plane[mask], nan=0.0))))
        vel_gauss.append(np.mean(float(les_dict['meanU'][0,0,0]) * (1 - np.nan_to_num(ga_plane[mask], nan=0.0))))
    vel_les = np.array(vel_les)
    vel_noj = np.array(vel_noj)
    vel_gauss = np.array(vel_gauss)

    # If requested, map velocities to absolute power via power curve
    if use_powercurve:
        try:
            wind_curve, power_curve = read_power_curve()
            P_les = power_from_power_curve(vel_les, wind_curve, power_curve)
            P_noj = power_from_power_curve(vel_noj, wind_curve, power_curve)
            P_gauss = power_from_power_curve(vel_gauss, wind_curve, power_curve)
        except Exception:
            # fallback to cubic
            P_les = (vel_les / float(les_dict['meanU'][0,0,0]))**3
            P_noj = (vel_noj / float(les_dict['meanU'][0,0,0]))**3
            P_gauss = (vel_gauss / float(les_dict['meanU'][0,0,0]))**3
    else:
        P_les = (vel_les / float(les_dict['meanU'][0,0,0]))**3
        P_noj = (vel_noj / float(les_dict['meanU'][0,0,0]))**3
        P_gauss = (vel_gauss / float(les_dict['meanU'][0,0,0]))**3

    return {'x': np.array(x_values), 'vel_les': vel_les, 'vel_noj': vel_noj, 'vel_gauss': vel_gauss,
            'P_les': P_les, 'P_noj': P_noj, 'P_gauss': P_gauss}


def plot_power_results(res_dict):
    x = res_dict['x']
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(x, res_dict['vel_les'], 'o-', label='LES vel', color='tab:green')
    ax1.plot(x, res_dict['vel_noj'], 's-', label='NOJ vel', color='tab:blue')
    ax1.plot(x, res_dict['vel_gauss'], 'd-', label='Gaussian vel', color='tab:red')
    ax1.set_xlabel('Downstream x/R')
    ax1.set_ylabel('Rotor-averaged velocity [m/s]')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(x, res_dict['P_les'], 'o-.', color='tab:green', alpha=0.6, label='LES P')
    ax2.plot(x, res_dict['P_noj'], 's-.', color='tab:blue', alpha=0.6, label='NOJ P')
    ax2.plot(x, res_dict['P_gauss'], 'd-.', color='tab:red', alpha=0.6, label='Gauss P')
    ax2.set_ylabel('Power (absolute or normalized)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
    plt.title('Rotor-averaged velocity and power vs downstream distance')
    plt.grid()
    plt.show()


# Simple example usage (if run as script)
if __name__ == '__main__':
    tr_path = os.path.join(os.getcwd(), 'TemplateReport2_46211(1).py')
    TR = None
    if os.path.exists(tr_path):
        try:
            # temporarily suppress plt.show to avoid windows from imported script
            _real_show = plt.show
            plt.show = lambda *args, **kwargs: None
            spec = importlib.util.spec_from_file_location('TR', tr_path)
            TR = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(TR)
            plt.show = _real_show
        except Exception as e:
            # restore show in case of exception
            try:
                plt.show = _real_show
            except Exception:
                pass
            print(f'Failed to import TemplateReport by path: {e}')
            TR = None
    if TR is not None:
        # Use objects from the imported report if available
        les_dict = TR.les
        noj_def = TR.noj['Ud']
        gauss_def = TR.gauss['Ud']
        x_vals = np.linspace(2,20,10)
        res = estimate_power_with_powercurve(les_dict, noj_def, gauss_def, x_vals, rotor_radius=1.0)
        # save the power plot to file instead of showing interactive window
        plot_power_results(res)
        try:
            figpath = os.path.join(os.getcwd(), 'power_vs_distance.png')
            plt.savefig(figpath, dpi=200, bbox_inches='tight')
            print('Saved power figure to', figpath)
        except Exception as e:
            print('Failed to save figure:', e)
    else:
        print('power_production.py: Import failed - please call functions from your running TemplateReport script and pass les/noj/gauss arrays.')
'''