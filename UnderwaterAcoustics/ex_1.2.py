import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so 'acoustics' package can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoustics.gsw_sound_speed import gsw_sound_speed
from acoustics.SS_CTD import mackenzie_sound

# Define regions with their coordinates
regions = {
    'Pacific Ocean': {'lat': 15.0, 'lon': 195.0},
    'Atlantic Ocean': {'lat': 40.0, 'lon': -40.0},
    'North Sea': {'lat': 56.0, 'lon': 5.0},
    'Baltic Sea': {'lat': 57.5, 'lon': 17.5}
}

# Assumed two-way travel time to seafloor (seconds)
# You would normally measure this with an echosounder
TWT = 3.0  # example: 3 seconds

def compute_depth_from_sound_speed(sound_speed, depth_levels):
    """
    Compute seafloor depth from two-way travel time and sound speed profile.
    
    Parameters:
    -----------
    sound_speed : array
        Sound speed profile (m/s)
    depth_levels : array
        Depth levels (m)
    
    Returns:
    --------
    float : Estimated seafloor depth (m)
    """
    # For each depth layer, compute time to travel through it
    # Time = distance / speed
    # For two-way travel: TWT = 2 * sum(layer_thickness / sound_speed)
    
    # Create depth intervals
    dz = np.diff(depth_levels)
    
    # Average sound speed in each layer
    c_avg = (sound_speed[:-1] + sound_speed[1:]) / 2
    
    # Two-way travel time through each layer
    twt_per_layer = 2 * dz / c_avg
    
    # Cumulative two-way travel time
    cumulative_twt = np.cumsum(twt_per_layer)
    
    # Find depth corresponding to target TWT
    # Interpolate to find exact depth for given TWT
    if TWT > cumulative_twt[-1]:
        print(f"Warning: TWT ({TWT}s) exceeds profile depth")
        return depth_levels[-1]
    
    estimated_depth = np.interp(TWT, cumulative_twt, depth_levels[1:])
    return estimated_depth

# Load oceanographic data
# NOTE: Replace this with your Copernicus Marine data when available
data_dir = Path(__file__).parent.parent / "data"
ds = xr.load_dataset(data_dir / "NA_TS_P1D-d230605.nc")

results = {}

print("=" * 80)
print("ACOUSTIC DEPTH ESTIMATION - Regional Comparison")
print("=" * 80)
print(f"Two-Way Travel Time (TWT): {TWT} seconds\n")

for region_name, coords in regions.items():
    print(f"\n{'='*80}")
    print(f"Region: {region_name}")
    print(f"Location: {coords['lat']}°N, {coords['lon']}°E")
    print(f"{'='*80}")
    
    # Extract temperature and salinity profiles at the region's location
    try:
        T = ds.thetao.isel(time=0).sel(latitude=coords['lat'], longitude=coords['lon'], method="nearest")
        S = ds.so.isel(time=0).sel(latitude=coords['lat'], longitude=coords['lon'], method="nearest")
        depth = ds.depth
        
        # Remove NaN values
        mask = ~(np.isnan(T.values) | np.isnan(S.values))
        T_clean = T.values[mask]
        S_clean = S.values[mask]
        depth_clean = depth.values[mask]
        pressure = depth_clean  # Approximate: 1 dbar ≈ 1 m
        
        if len(T_clean) < 2:
            print(f"  ⚠ Insufficient data for {region_name}")
            continue
        
        # Compute sound speed using both methods
        gsw_ss = gsw_sound_speed(S_clean, T_clean, pressure)
        mackenzie_ss = mackenzie_sound(T_clean, S_clean, depth_clean)
        
        # Compute depth estimates
        depth_gsw = compute_depth_from_sound_speed(gsw_ss, depth_clean)
        depth_mackenzie = compute_depth_from_sound_speed(mackenzie_ss, depth_clean)
        
        # Calculate error
        depth_error = depth_mackenzie - depth_gsw
        relative_error = (depth_error / depth_gsw) * 100
        
        # Store results
        results[region_name] = {
            'depth_gsw': depth_gsw,
            'depth_mackenzie': depth_mackenzie,
            'error_m': depth_error,
            'error_pct': relative_error,
            'gsw_ss': gsw_ss,
            'mackenzie_ss': mackenzie_ss,
            'depth_levels': depth_clean,
            'T': T_clean,
            'S': S_clean
        }
        
        # Display results
        print(f"\n  TEOS-10 (GSW) depth:      {depth_gsw:8.2f} m")
        print(f"  Mackenzie depth:          {depth_mackenzie:8.2f} m")
        print(f"  Absolute error:           {depth_error:8.2f} m")
        print(f"  Relative error:           {relative_error:8.4f} %")
        
    except Exception as e:
        print(f"  ⚠ Error processing {region_name}: {e}")

# Visualization
if results:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Acoustic Depth Estimation - Regional Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (region_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot sound speed profiles
        ax.plot(data['gsw_ss'], data['depth_levels'], 'b-', linewidth=2, label='TEOS-10')
        ax.plot(data['mackenzie_ss'], data['depth_levels'], 'r--', linewidth=2, label='Mackenzie')
        
        # Mark estimated depths
        ax.axhline(y=data['depth_gsw'], color='b', linestyle=':', alpha=0.7, 
                   label=f'GSW: {data["depth_gsw"]:.1f}m')
        ax.axhline(y=data['depth_mackenzie'], color='r', linestyle=':', alpha=0.7,
                   label=f'Mack: {data["depth_mackenzie"]:.1f}m')
        
        ax.set_xlabel('Sound Speed (m/s)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.set_title(f'{region_name}\nError: {data["error_m"]:.2f}m ({data["error_pct"]:.3f}%)', 
                     fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Region':<20} {'TEOS-10 (m)':<15} {'Mackenzie (m)':<15} {'Error (m)':<12} {'Error (%)':<12}")
    print("-"*80)
    for region_name, data in results.items():
        print(f"{region_name:<20} {data['depth_gsw']:>14.2f} {data['depth_mackenzie']:>14.2f} "
              f"{data['error_m']:>11.2f} {data['error_pct']:>11.4f}")
    print("="*80)

else:
    print("\n⚠ No results to display. Please check your data files.")

print("\nNOTE: This uses sample data. Replace with Copernicus Marine data for your actual regions.")
