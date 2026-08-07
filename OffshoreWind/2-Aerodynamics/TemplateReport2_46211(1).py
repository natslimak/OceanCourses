import numpy as np
import netCDF4 as nc
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This imports the 3D plotting toolkit
from scipy.optimize import curve_fit
import ExcelFunctions
import pandas as pd


# NOJ wake model by Jensen, 1983
# NOJensen wake model function
def nojensen(param, mesh):
    # N. O. Jensen's equation rescaled to [R]
    k = param[0]
    CT = param[1]
    return (1 - np.sqrt(1 - CT)) / ((1 + (k * mesh['x'])) ** 2)

def gaussian(param, mesh):
    k = param[0]
    CT = param[1]
    eps = param[2]
    zh=170
    delta_U = (1-np.sqrt(1-(CT/(8*(k*mesh['x']/2+eps)**2))))*np.exp((-1/(2*(k*mesh['x']/2+eps)**2))*((mesh['y']/2)**2))
    #We do the calculation at z=zh so this term =0
    return delta_U

# INPUT
loaddir = './'
scenario = '46211_LES_data'

# Loading LES
ncinf = nc.Dataset(f'{loaddir}{scenario}.nc')

# Load all variables from the NetCDF file into a dictionary
les = {}
for var in ncinf.variables:
    les[var] = ncinf.variables[var][:]

# Transposing to have x, y, z, t as 1st, 2nd, 3rd, 4th dimensions:
les['x'] = np.transpose(les['x'], (2, 1, 0))
les['y'] = np.transpose(les['y'], (2, 1, 0))
les['z'] = np.transpose(les['z'], (2, 1, 0))
les['U'] = np.transpose(les['U'], (3, 2, 1, 0))

les['meanU'] = np.mean(les['U'], axis=3)
les['stdU'] = np.std(les['U'], axis=3)

index_y=0
while les['y'][0,index_y,0]<0:
    index_y+=1

index_z=0
while les['z'][0,0,index_z]<0:
    index_z+=1

print('i=',index_y, 'j=',index_z)
mean_inflow_vel=les['meanU'][0,index_y,index_z] #inflow for x=-5, y=0, z=0 and t=0s
print('Mean inflow velocity = ', mean_inflow_vel)

# Calculate turbulence intensity from mean and std
turb_intensity=les['stdU']/les['meanU']
mean_turb=turb_intensity[0,index_y,index_z]*100
print('Mean turbulence=',mean_turb,'%')

# NOJ input parameters
noj = {}
noj['k'] = 0.05  # entrainment constant

# Can be taken now from ExcelFunctions.py using interpolate_ct function
# CT = ExcelFunctions.interpolate_ct(mean_inflow_vel)

CT = 0.855257296 #from the excel with wind speed = 8.4m
rotor_speed = 5.232824601 #rad/s from the excel
rotor_d = 284.0 #m from the report
TSR= rotor_speed*rotor_d/(2*mean_inflow_vel)
#print('>>>> Remember to update CT')
param = [noj['k'], CT, TSR]
print("TSR =", TSR)

#Gaussian imput parameters
gauss={}
gauss['k']=0.031
beta=1/2*(1+np.sqrt(1-CT))/np.sqrt(1-CT)
eps=0.2*np.sqrt(beta)
gauss_param=[gauss['k'],CT,eps]
print("gauss param",gauss_param)


# Calculate deficit
noj['Ud'] = nojensen(param, les)
gauss['Ud'] = gaussian(gauss_param, les)
#print(gauss['Ud'].shape)


# Setting all values outside of linear wake expansion to 0
for i in range(les['x'].shape[0]):
    # Create a boolean mask to identify points inside the wake expansion
    idx = np.sqrt(les['y'][i,:,:]**2 + les['z'][i,:,:]**2) <= param[0] * les['x'][i,0,0] + 1
    
    # Apply the mask to zero out values outside the wake expansion
    noj['Ud'][i,:,:] = idx * noj['Ud'][i,:,:]


noj_result = noj['Ud'][1:, index_y, index_z]  
gauss_result = gauss['Ud'][1:,index_y,index_z]

# Centerline x values (needed to be in R)
les_result = 1 - les['meanU'][1:,index_y,index_z]/mean_inflow_vel
x_centerline = les['x'][1:, index_y, index_z]


plt.plot(x_centerline, noj_result,color='blue',label='NOJensen(k=0.05)')
plt.plot(x_centerline, gauss_result, color = 'red',label='Gaussian(k=0.031)')
plt.plot(x_centerline,les_result,color='green', marker='o', linestyle='None', label='LES')
plt.xlabel('Downstream distance x/R')
plt.ylabel('Centerline wake deficit')
plt.grid()
plt.legend()


# Assuming les.x, les.y, les.z, and les.meanU are numpy arrays or lists
x = les['x'].flatten()  # Equivalent to MATLAB's les.x(:)
y = les['y'].flatten()  # Equivalent to MATLAB's les.y(:)
z = les['z'].flatten()  # Equivalent to MATLAB's les.z(:)
meanU = les['meanU'].flatten()  # Equivalent to MATLAB's les.meanU(:)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color and size determined by meanU
scatter = ax.scatter(x, y, z, c=meanU, s=meanU, cmap='viridis')

# Adding labels
ax.set_xlabel('x [R]')
ax.set_ylabel('y [R]')
ax.set_zlabel('z [R]')

# Display the plot
plt.colorbar(scatter)  # Adding a color bar to represent meanU values
plt.show()



# --- Calibration and comparison plot added below ---

def optimize_and_plot(centerline_start, centerline_end):
    # Calibration region
    calib_mask = (x_centerline >= centerline_start) & (x_centerline <= centerline_end)
    calib_x = x_centerline[calib_mask]
    calib_les = les_result[calib_mask]

    # NOJ fit function
    def nojensen_fit(x, k):
        CT = param[1]
        return (1 - np.sqrt(1 - CT)) / ((1 + (k * x)) ** 2)

    popt_noj, _ = curve_fit(nojensen_fit, calib_x, calib_les, p0=[0.05])
    calib_k_noj = popt_noj[0]
    print(f"Calibrated NOJ k: {calib_k_noj}")

    # Gaussian fit function
    def gaussian_fit(x, k):
        CT = param[1]
        beta = 1/2*(1+np.sqrt(1-CT))/np.sqrt(1-CT)
        eps = 0.2*np.sqrt(beta)
        y = 0  # centerline
        return (1-np.sqrt(1-(CT/(8*(k*x/2+eps)**2))))*np.exp((-1/(2*(k*x/2+eps)**2))*((y/2)**2))

    try:
        popt_gauss, _ = curve_fit(gaussian_fit, calib_x, calib_les, p0=[0.031])
        calib_k_gauss = popt_gauss[0]
        print(f"Calibrated Gaussian k: {calib_k_gauss}")
    except Exception as e:
        print(f"curve_fit failed for Gaussian: {e}. Using initial k value.")
        calib_k_gauss = gauss_param[0]

    # Recalculate deficits with calibrated parameters
    noj_calib_param = [calib_k_noj, param[1], TSR]
    gauss_calib_param = [calib_k_gauss, param[1], 0.2*np.sqrt(1/2*(1+np.sqrt(1-param[1]))/np.sqrt(1-param[1]))]
    noj_calib = nojensen(noj_calib_param, les)
    gauss_calib = gaussian(gauss_calib_param, les)

    # Plot calibrated models and mark calibration points
    plt.figure()
    #NOJensen
    plt.plot(x_centerline, noj_result, color='blue', linestyle=':', label='NOJensen (k=0.05)')
    plt.plot(x_centerline, noj_calib[1:, index_y, index_z], color='blue',label=f'Optimized NOJensen (k={calib_k_noj:.3f})')

    #Gaussian
    plt.plot(x_centerline, gauss_result, color='red',  linestyle=':',label='Gaussian (k=0.031)')
    plt.plot(x_centerline, gauss_calib[1:, index_y, index_z], color='red', label=f'Optimized Gaussian (k={calib_k_gauss:.3f})')

    #LES
    plt.plot(x_centerline, les_result,color='green', marker='o', linestyle='None', label='LES')
    plt.plot(calib_x, calib_les, color='green', marker='s', linestyle='None', label='LES calibration points')
    
    plt.xlabel('Downstream distance x/R')
    plt.ylabel('Centerline wake deficit')
    plt.xlim(0, 20)
    plt.legend()
    plt.title('Centerline Wake Deficit: Calibrated Models vs LES')
    plt.grid()
    plt.show()
    return calib_k_noj, calib_k_gauss, noj_calib, gauss_calib

# Optimized nojensen and gaussian models
#optimize_and_plot(1, 20)
#optimize_and_plot(5, 20)
#optimize_and_plot(10, 20)


# --- Lateral wake profile plotting function added below ---

def plot_lateral_profiles(x_positions=[2,6,12,20], z_index=None, show=True):
    """Plot lateral (y) wake deficit profiles at specified downstream x (in R).
    Plots LES (points), NOJensen and Gaussian model curves for each x in a 1xN subplot layout.
    Returns a dict of RMSE values per x for both models.
    """
    
    calib_k_noj, calib_k_gauss, noj_calib, gauss_calib =optimize_and_plot(10, 20)

    if z_index is None:
        z_index = index_z

    n = len(x_positions)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    if n == 1:
        axs = [axs]

    from math import isfinite
    def rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            return np.nan
        return np.sqrt(np.mean((a[mask] - b[mask])**2))

    rmses = {} # Root Mean Square Error used to measure differences between values predicted by a model and the values actually observed
    for ax, x_val in zip(axs, x_positions):
        # find nearest x index
        xi = (np.abs(les['x'][:,0,0] - x_val)).argmin()
        y_coords = les['y'][xi, :, z_index]

        # LES profile (deficit)
        les_prof = 1.0 - les['meanU'][xi, :, z_index] / mean_inflow_vel
        # model profiles (already computed arrays)
        noj_prof = noj_calib[xi, :, z_index]
        gauss_prof = gauss_calib[xi, :, z_index]
        #noj_prof = noj['Ud'][xi, :, z_index]
        #gauss_prof = gauss['Ud'][xi, :, z_index]


        # Sort by y for clean plotting (in case ordering isn't monotonic)
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        les_sorted = les_prof[sort_idx]
        noj_sorted = noj_prof[sort_idx]
        gauss_sorted = gauss_prof[sort_idx]

        # plot
        ax.plot(y_sorted, noj_sorted, '-', color='tab:blue', label='NOJensen')
        ax.plot(y_sorted, gauss_sorted, '--', color='tab:red', label='Gaussian')
        ax.plot(y_sorted, les_sorted, 'o', color='tab:green', label='LES')
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(f'x = {les['x'][xi,0,0]:.1f} R')
        ax.set_xlabel('y/R')
        ax.grid(True)

        # compute RMSEs
        rms_noj = rmse(noj_sorted, les_sorted)
        rms_gauss = rmse(gauss_sorted, les_sorted)
        rmses[x_val] = {'NOJ_rmse': rms_noj, 'GAUSS_rmse': rms_gauss}

        # annotate RMSE on subplot
        ax.text(0.02, 0.95, f'NOJ RMSE={rms_noj:.3f}\nGAUSS RMSE={rms_gauss:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    axs[0].set_ylabel('Velocity deficit')
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('Lateral wake profiles at selected downstream locations')
    plt.tight_layout(rect=[0,0,0.95,0.95])

    if show:
        plt.show()
    return rmses


# Example call (uncomment to run)
rmses = plot_lateral_profiles([2,6,12,20])
print('Per-location RMSEs:', rmses)


'''
# --- Plot only the plane in front of the turbine ---
# Find the index where x = -6 (or closest value)
x_plane_val = -6
x_plane_idx = (np.abs(les['x'][:,0,0] - x_plane_val)).argmin()

# Extract y, z, meanU for this plane
plane_y = les['y'][x_plane_idx,:,:]
plane_z = les['z'][x_plane_idx,:,:]
plane_meanU = les['meanU'][x_plane_idx,:,:]

plt.figure()
plt.pcolormesh(plane_y, plane_z, plane_meanU, cmap='viridis', shading='auto')
plt.xlabel('y [R]')
plt.ylabel('z [R]')
plt.title(f'MeanU at x = {les['x'][x_plane_idx,0,0]:.2f}')
plt.colorbar(label='MeanU')
plt.grid()
plt.show()
'''

# --- Power production for potential second turbine ---

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


# ----------------------------------------------------------------------------
wind_curve, power_curve, power_coeff_curve = read_power_curve() 


def plot_mean_minus_les_deficit(mean_vel, x_max=20, show=True):
    """Compute LES centerline deficit and plot mean_vel minus the LES deficit (scaled) over 0..x_max (R).

    Interpretation: LES centerline deficit is defined as d(x) = 1 - meanU(x)/mean_vel.
    We compute velocity_est(x) = mean_vel * (1 - d(x)) == meanU(x) (exact LES mean), and
    alt_est(x) = mean_vel - d(x) (pure subtraction, kept for comparison) — units may be inconsistent.

    Returns: (x_vals, velocity_est, alt_est)
    """
    # get centerline indices and arrays (use existing x_centerline and les_result if available)
    try:
        x_vals = les['x'][1:, index_y, index_z]
        les_deficit = 1.0 - les['meanU'][1:, index_y, index_z] / mean_vel
    except Exception:
        # fallback: compute from previously derived variables
        x_vals = x_centerline
        les_deficit = les_result

    # limit to x_max
    mask = x_vals <= x_max
    x_plot = x_vals[mask]
    d_plot = les_deficit[mask]

    # velocity estimates
    velocity_est = mean_vel * (1.0 - d_plot)  # equals LES meanU on centerline
    alt_est = mean_vel*(1-d_plot)              # direct subtraction (less physically meaningful)

    # Plotting the velocities with calibrated models
    calib_k_noj, calib_k_gauss, noj_calib, gauss_calib= optimize_and_plot(10, 20) # For 10R to 20R calibration
    # Calibrated NOJensen velocity
    noj_calib_center=noj_calib[1:,index_y,index_z]
    nojensen_deficit_calib = noj_calib_center[mask]
    velocity_nojens_calib = mean_vel*(1-nojensen_deficit_calib)
    # Calibrated Gaussian velocity
    gauss_calib_center=gauss_calib[1:,index_y,index_z]
    gaussian_deficit_calib = gauss_calib_center[mask]
    velocity_gauss_calib = mean_vel *(1- gaussian_deficit_calib)

    plt.figure()
    #plt.plot(x_plot, velocity_est, '-', color='tab:blue', label='mean_vel * (1 - LES_deficit) (LES meanU)')
    plt.plot(x_plot, mean_vel * np.ones_like(x_plot), ':', color='k', label='mean_vel (inflow)')
    plt.plot(x_plot, alt_est, '--', color='tab:orange', label='mean_vel*(1-LES)')
    #plt.plot(x_plot, velocity_nojens, '-.', color='tab:purple', label='mean_vel - NOJensen_deficit')
    plt.plot(x_plot, velocity_nojens_calib, '-', color='tab:red', label='mean_vel*(1-NOJ_calibrated)')
    plt.plot(x_plot, velocity_gauss_calib, '-', color='tab:green', label='mean_vel*(1-Gaussian_calibrated)')
    plt.xlabel('Downstream distance x/R')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity in the wake (centerline)')
    plt.legend()
    plt.grid()
    plt.show()
    return x_plot, velocity_est, alt_est, velocity_nojens_calib, velocity_gauss_calib


def plot_power_from_deficit(mean_vel, x_max=20, use_alt=True, show=True):
    """Interpolate velocities from the mean-minus-deficit arrays to the turbine power curve and plot power vs x/R.

    Parameters:
        mean_vel: reference mean inflow velocity (m/s)
        x_max: maximum downstream distance in rotor radii to plot
        use_alt: if True use alt_est (mean_vel - deficit), otherwise use velocity_est (mean_vel*(1-deficit))
        show: whether to display the plot interactively
    Returns:
        x_plot, P  (arrays)
    """
    # get velocity arrays from the existing function (don't show its plot)
    x_plot, velocity_est, alt_est, velocity_nojens_calib, velocity_gauss_calib = plot_mean_minus_les_deficit(mean_vel, x_max=x_max, show=False)
    vel_arr = alt_est
    vel_arr_jensen = velocity_nojens_calib 
    vel_arr_gauss = velocity_gauss_calib 

    # interpolate to power using the rotor performance curve read earlier
    P = np.interp(vel_arr, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])
    P_jensen = np.interp(vel_arr_jensen, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])
    P_gauss = np.interp(vel_arr_gauss, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])

    plt.figure()
    plt.plot(x_plot, P, '--', color='tab:orange', label='Estimated power (from Excel power curve)')
    plt.plot(x_plot, P_jensen, '-', color='tab:red', label='Estimated power (NOJensen calibrated)')
    plt.plot(x_plot[1:], P_gauss[1:], '-', color='tab:green', label='Estimated power (Gaussian calibrated)')
    plt.xlabel('Downstream distance x/R')
    plt.ylabel('Power [MW]')
    plt.title('Estimated power from LES-derived velocity')
    plt.grid()
    plt.legend()
    if show:
        plt.show()
    return x_plot, P


# Produce the power plot using alt_est (raw subtraction) by default
#plot_mean_minus_les_deficit(mean_vel, x_max=20, show=True)
plot_power_from_deficit(mean_inflow_vel, x_max=20, use_alt=True, show=True)



from scipy.interpolate import LinearNDInterpolator
#Creating a polar grid
r = np.linspace(0, 1, 100)   # rayon
theta = np.linspace(0, 2*np.pi, 100)  # angle

R, Theta = np.meshgrid(r, theta)

#Initial cartesian coordinates
y = les['y'][0,:,:].ravel()
z = les['z'][0,:,:].ravel()

# Converte into cartesian coordinate for interpolation
Y_pol = R * np.cos(Theta)
Z_pol = R * np.sin(Theta)

def interpolate_to_polar(method,y,z,Y_pol,Z_pol,mean_pol_U):
    if method==1: velocities=les['meanU']
    elif method ==2: velocities=noj_calib
    elif method==3 : velocities=gauss_calib
    else: return('wrong method')

    Pol_U=[]; Mean_Vel=[]
    for i in range(len(les['x'][:,0,0])):
        #Caluclates the velocities to interpolate
        values=velocities[i,:,:].ravel()
        #caluclate the interpolated U 
        F = LinearNDInterpolator((y,z), values)
        pol_U=F(Y_pol,Z_pol)
        Pol_U.append(pol_U)

        if i==0 and method==1 : mean_pol_U=np.mean(pol_U)
        elif method == 1 :
            mean_pol_vel=np.mean(pol_U)
            Mean_Vel.append(mean_pol_vel)
        else:
            real_vel=mean_pol_U*(1-pol_U)
            mean_pol_vel=np.mean(real_vel)
            Mean_Vel.append(mean_pol_vel)

    return Pol_U,Mean_Vel, mean_pol_U

pol_U_les,mean_vel_les, mean_pol_U = interpolate_to_polar(1,y,z,Y_pol,Z_pol,0)

calib_k_noj, calib_k_gauss, noj_calib, gauss_calib =optimize_and_plot(10, 20)
pol_U_noj,mean_vel_noj, mean_pol_U = interpolate_to_polar(2,y,z,Y_pol,Z_pol,mean_pol_U)
pol_U_gauss,mean_vel_gauss, mean_pol_U = interpolate_to_polar(3,y,z,Y_pol,Z_pol,mean_pol_U)



plt.figure()
#plt.plot(x_plot, velocity_est, '-', color='tab:blue', label='mean_vel * (1 - LES_deficit) (LES meanU)')
plt.plot(x_centerline, [mean_pol_U]*len(x_centerline), ':', color='k', label='mean_vel (inflow)')
plt.plot(x_centerline, mean_vel_les, '--', color='tab:orange', label='mean_vel*(1-LES)')
#plt.plot(x_plot, velocity_nojens, '-.', color='tab:purple', label='mean_vel - NOJensen_deficit')
plt.plot(x_centerline, mean_vel_noj[1:], '-', color='tab:red', label='mean_vel*(1-NOJensen_calibrated)')
plt.plot(x_centerline[1:], mean_vel_gauss[2:], '-', color='tab:green', label='mean_vel*(1-Gaussian_calibrated)')
plt.xlabel('Downstream distance x/R')
plt.ylabel('Velocity [m/s]')
plt.title('Mean velocities in the wake (in polar coordinate)')
plt.legend()
plt.grid()
plt.show()

#Plot the power
wind_curve, power_curve, power_coeff_curve = read_power_curve()
P_polar_les = np.interp(mean_vel_les, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])
P_polar_noj = np.interp(mean_vel_noj, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])
P_polar_gauss = np.interp(mean_vel_gauss, wind_curve, power_curve, left=power_curve[0], right=power_curve[-1])

plt.figure()
#plt.plot(x_plot, velocity_est, '-', color='tab:blue', label='mean_vel * (1 - LES_deficit) (LES meanU)')
#plt.plot(x_centerline, mean_pol_U, ':', color='k', label='mean_vel (inflow)')
plt.plot(x_centerline, P_polar_les, '--', color='tab:orange', label='mean_vel*(1-LES)')
#plt.plot(x_plot, velocity_nojens, '-.', color='tab:purple', label='mean_vel - NOJensen_deficit')
plt.plot(x_centerline, P_polar_noj[1:], '-', color='tab:red', label='mean_vel*(1-NOJensen_calibrated)')
plt.plot(x_centerline[1:],P_polar_gauss[2:], '-', color='tab:green', label='mean_vel*(1-Gaussian_calibrated)')
plt.xlabel('Downstream distance x/R')
plt.ylabel('Power [MW]')
plt.title('Estimated power in the wake (in polar coordinate)')
plt.legend()
plt.grid()
plt.show()

print('let us try one more time to push')
