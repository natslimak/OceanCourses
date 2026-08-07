import numpy as np
import pandas as pd

EXCEL_PATH = 'IEA-22-280-RWT_tabular.xlsx'
SHEET_NAME = 'Rotor Performance'
WIND_COL = 0  # Column A
CT_COL = 8    # Column I

def interpolate_ct(mean_inflow_vel):
    """
    Interpolates thrust coefficient (CT) from the fixed Excel file for a given mean inflow velocity.
    Args:
        mean_inflow_vel (float): The mean inflow velocity to interpolate for.
    Returns:
        float: Interpolated CT value.
    """
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    wind_speeds = df.iloc[:, WIND_COL].values
    thrust_coeffs = df.iloc[:, CT_COL].values
    idx = np.searchsorted(wind_speeds, mean_inflow_vel)
    if idx == 0:
        idxs = [0, 1]
    elif idx == len(wind_speeds):
        idxs = [-2, -1]
    else:
        idxs = [idx-1, idx]
    ws_low, ws_high = wind_speeds[idxs[0]], wind_speeds[idxs[1]]
    ct_low, ct_high = thrust_coeffs[idxs[0]], thrust_coeffs[idxs[1]]
    if ws_high != ws_low:
        ct_interp = ct_low + (ct_high - ct_low) * (mean_inflow_vel - ws_low) / (ws_high - ws_low)
    else:
        ct_interp = ct_low
    return ct_interp
