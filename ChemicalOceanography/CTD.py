import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load the CSV file
data_path = Path(__file__).resolve().parent / 'Data' / 'ChemOcean26.csv'
df = pd.read_csv(data_path)

# Group the data by the location (lat and long) and time
grouped = df.groupby(['lat', 'long', 'time'])
grouped = grouped.apply(lambda x: x.reset_index(drop=True))


# Function to plot profiles for a specific day
def plot_profiles_by_day(df, day_prefix):
    day_df = df[df['time'].astype(str).str.startswith(day_prefix)]
    day_grouped = day_df.groupby(['lat', 'long', 'time'])

    profiles = [
        ('Salinity', 'Salinity', 'PSU', 'tab:blue'),
        ('Temperature', 'Temperature', '°C', 'tab:orange'),
        ('Density', 'Density', 'kg/m³', 'tab:green'),
        ('Oxygen', 'Oxygen', 'µmol/kg', 'tab:red'),
        ('Turbidity', 'Turbidity', 'NTU', 'tab:purple'),
        ('ChlF', 'ChlF', 'mg/m³', 'tab:brown'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(8, 12), sharey=True)
    axes = axes.ravel()
    fig.suptitle(f'CTD Profiles for {day_prefix}', fontsize=16)

    for (lat, long, time), group in day_grouped:
        depth = group['Depth'].values
        for ax, (column, title, units, color) in zip(axes, profiles):
            ax.plot(group[column].values, depth, color=color, alpha=0.5)
            ax.set_title(f'{title}')
            ax.set_xlabel(f'{title} ({units})')
            ax.invert_yaxis()

    axes[0].set_ylabel('Depth')
    axes[3].set_ylabel('Depth')
    axes[0].invert_yaxis()
    fig.tight_layout()
    plt.show()

def plot_map(grouped):
    plt.figure(figsize=(10, 6))
    for (lat, long, time), group in grouped:
        plt.scatter(long, lat, label=f'{time}', s=100)  # Plotting the location
    plt.title('CTD Profiles Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.legend()
    plt.grid()
    plt.show()

# Main function to execute the plotting
def main():
    plot_map(grouped)
    plot_profiles_by_day(df, '18-Apr-2026')
    plot_profiles_by_day(df, '19-Apr-2026')
    count_transects = 1
    for (lat, long, time), group in grouped:
        print(f'{count_transects}. Location: ({lat:.4f}, {long:.4f}, {time}), Depth range: {group["Depth"].min()} - {group["Depth"].max()} m')
        count_transects += 1

if __name__ == "__main__":
    main()