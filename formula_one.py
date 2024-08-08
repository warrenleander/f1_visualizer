import fastf1
from fastf1 import plotting
from fastf1.plotting import get_driver_color
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from windrose import WindroseAxes

# Enable cache
try:
    fastf1.Cache.enable_cache(os.path.join(sys.path[0], "fastf1_cache"))
except:
    os.makedirs(os.path.join(sys.path[0], "fastf1_cache"))
    fastf1.Cache.enable_cache(os.path.join(sys.path[0], "fastf1_cache"))

# Input session and year
year = 2023
location = 'Singapore'
session = 'R'  # Can be 'Q', 'FP1', 'FP2', 'FP3', or 'R'

# Get session
race = fastf1.get_session(year, location, session)
race.load(weather=True)

# Load race laps
race_name = race.event.OfficialEventName
df = race.laps

# Process dataframe
if session == 'R':
    df = df.sort_values(by=['Position', 'LapNumber'], ascending=[True, False]).reset_index(drop=True)
else:
    df = df[~df['Deleted']].sort_values(by=['LapNumber','Position'], ascending=[False, True]).reset_index(drop=True)

# Convert time columns to seconds
for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
    df[col] = df[col].dt.total_seconds()

# Fill in empty laptime records
df['LapTime'] = df['LapTime'].fillna(df['Sector1Time'] + df['Sector2Time'] + df['Sector3Time'])

# Weather data
df_weather = race.weather_data.copy()
df_weather['SessionTime(Minutes)'] = df_weather['Time'].dt.total_seconds() / 60

# Rain Indicator
rain = df_weather.Rainfall.any()

# Now create the driver_color dictionary
driver_color = {row['Driver']: get_driver_color(row['Driver'], session=race) for _, row in df.iterrows()}

# Get results sorted by final position for race
if session == 'R':
    df_results = race.results[['Position', 'Abbreviation', 'TeamName', 'Status', 'Time']]
    df_results['TimeDiff'] = pd.to_timedelta(df_results['Time']).dt.total_seconds()
    df_results.loc[df_results['Position'] == 1, 'TimeDiff'] = 0
    df_results['TimeDiff'] = df_results['TimeDiff'].fillna(float('inf'))  # Handle DNFs
else:
    df_results = df.groupby('Driver')['LapTime'].min().sort_values().reset_index()
    df_results['Position'] = df_results.index + 1
    df_results['TimeDiff'] = df_results['LapTime'] - df_results['LapTime'].min()

# Add Abbreviation column to df_results if not present
if 'Abbreviation' not in df_results.columns:
    def get_driver_abbreviation(driver):
        try:
            return fastf1.api.driver_info(driver)['Abbreviation']
        except:
            return driver[:3].upper()
    df_results['Abbreviation'] = df_results['Driver'].apply(get_driver_abbreviation)

# Sort results
df_results = df_results.sort_values('Position').reset_index(drop=True)

# Get top 3 drivers
top_3_drivers = df_results.head(3)['Abbreviation'].tolist()

# Get telemetry data for top 3 drivers
telemetry_data = []
for driver in top_3_drivers:
    if session == 'R':
        driver_lap = df.loc[df.pick_driver(driver)['LapTime'].idxmin()]
    else:
        driver_lap = df.loc[df.pick_driver(driver)['LapTime'].idxmin()]
    tel = driver_lap.get_telemetry()
    tel['Driver'] = driver
    telemetry_data.append(tel)

combined_tel = pd.concat(telemetry_data)

# Calculate track performance based on fastest times for top 3
distance_step = 10  # meters
max_distance = combined_tel['Distance'].max()
distance_bins = np.arange(0, max_distance, distance_step)

track_performance = {driver: 0 for driver in top_3_drivers}
for bin_start in distance_bins[:-1]:
    bin_end = bin_start + distance_step
    bin_data = combined_tel[(combined_tel['Distance'] >= bin_start) & (combined_tel['Distance'] < bin_end)]
    if not bin_data.empty:
        fastest_driver = bin_data.loc[bin_data['Speed'].idxmax(), 'Driver']
        if isinstance(fastest_driver, pd.Series):
            fastest_driver = fastest_driver.iloc[0]
        track_performance[fastest_driver] += 1

total_bins = sum(track_performance.values())
track_performance = {k: v / total_bins * 100 for k, v in track_performance.items()}

# Get track layout from telemetry data of the pole position driver or race winner
fastest_driver = df_results.Abbreviation[0]
fastest_lap = df.pick_driver(fastest_driver).pick_fastest()
lap_tel = fastest_lap.get_telemetry()
lap_x, lap_y = lap_tel['X'], lap_tel['Y']

# Track Performance plot
plt.figure(figsize=(12, 10))
plt.title(f'Track Performance of Top 3 Drivers\n{race_name} - {session}', fontsize=20)
plt.xlabel('On Fastest Lap of Top 3 Drivers', fontsize=12)

# Create segments and colors for each segment
points = np.array([lap_x, lap_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Determine color for each segment based on fastest driver among top 3
segment_colors = []
for i in range(len(segments)):
    distance = lap_tel['Distance'].iloc[i]
    bin_index = int(distance // distance_step)
    if bin_index < len(distance_bins) - 1:
        bin_start = distance_bins[bin_index]
        bin_end = distance_bins[bin_index + 1]
        bin_data = combined_tel[(combined_tel['Distance'] >= bin_start) & (combined_tel['Distance'] < bin_end)]
        if not bin_data.empty:
            fastest_driver = bin_data.loc[bin_data['Speed'].idxmax(), 'Driver']
            if isinstance(fastest_driver, pd.Series):
                fastest_driver = fastest_driver.iloc[0]
            segment_colors.append(driver_color.get(fastest_driver, '#333333'))
    else:
        segment_colors.append('#333333')  # Default color for any out-of-range segments

# Create the colored line collection
lc = LineCollection(segments, colors=segment_colors, linewidths=2, zorder=10)
plt.gca().add_collection(lc)

# Add direction arrow
arrow_scale = 0.05
arrow_pos = int(len(lap_x) * 0.02)
dx = lap_x.iloc[arrow_pos] - lap_x.iloc[0]
dy = lap_y.iloc[arrow_pos] - lap_y.iloc[0]
plt.arrow(lap_x.iloc[0], lap_y.iloc[0], dx*arrow_scale, dy*arrow_scale, 
          head_width=100, head_length=200, fc='k', ec='k', zorder=20)

plt.axis('equal')
plt.axis('off')

# Create legend for top 3 drivers
legend_lines = [Line2D([0], [0], color=driver_color[driver], lw=4) for driver in top_3_drivers]
plt.legend(legend_lines,
           [f"{driver} | {track_performance[driver]:.1f}%" for driver in top_3_drivers],
           loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()

# Results plot
plt.figure(figsize=(12, 10))

# Filter out DNFs and create horizontal bar chart
df_results_plot = df_results[df_results['TimeDiff'] < float('inf')]
bars = plt.barh(df_results_plot['Abbreviation'], df_results_plot['TimeDiff'], 
                color=[driver_color.get(driver, '#333333') for driver in df_results_plot['Abbreviation']])

# Customize the plot
plt.xlabel('Gap to First' if session == 'R' else 'Gap to Fastest (seconds)')
plt.title(f'{session} Results\n{race_name}')
plt.gca().invert_yaxis()  # Invert y-axis to have the fastest driver at the top

# Add value labels on the bars
for i, v in enumerate(df_results_plot['TimeDiff']):
    plt.text(v, i, f' +{v:.3f}' if v > 0 else ' 0.000', va='center', fontweight='bold')

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add grid lines
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Sector plots
if session != 'R':
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, sector in zip(axes, ['Sector1Time', 'Sector2Time', 'Sector3Time']):
        top_10 = df.sort_values(sector).groupby('Driver').first().reset_index().sort_values(sector).head(10)
        sns.barplot(x=top_10[sector], y=top_10['Driver'], hue=top_10['Driver'], palette=driver_color, ax=ax, edgecolor='black', legend=False)
        ax.bar_label(ax.containers[0], padding=3)
        ax.set_xlim(top_10[sector].min()-0.1, top_10[sector].max()+0.1)
        ax.set_title(f'Sector {sector[6]}', fontweight="bold")

    plt.tight_layout()
    plt.show()

# Team speed analysis
team_speeds = df.groupby('Team').apply(lambda x: pd.Series({
    'Max Speed': x.pick_fastest().get_telemetry().Speed.max(),
    'Min Speed': x.pick_fastest().get_telemetry().Speed.min()
})).reset_index()

team_speeds = team_speeds.sort_values('Max Speed', ascending=False).reset_index(drop=True)

# Weather plots
fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)

# Track and Air Temperature
sns.lineplot(data=df_weather, x='SessionTime(Minutes)', y='TrackTemp', label='TrackTemp', ax=axes[0,0])
sns.lineplot(data=df_weather, x='SessionTime(Minutes)', y='AirTemp', label='AirTemp', ax=axes[0,0])
if rain:
    axes[0,0].fill_between(df_weather[df_weather.Rainfall]['SessionTime(Minutes)'], 
                         df_weather.TrackTemp.max()+0.5, df_weather.AirTemp.min()-0.5, 
                         facecolor="blue", color="blue", alpha=0.1, zorder=0, label='Rain')
axes[0,0].legend(loc='upper right')
axes[0,0].set_ylabel('Temperature')
axes[0,0].set_title('Track Temperature & Air Temperature (°C)')

# Humidity
sns.lineplot(df_weather, x='SessionTime(Minutes)', y='Humidity', ax=axes[0,1])
if rain:
    axes[0,1].fill_between(df_weather[df_weather.Rainfall]['SessionTime(Minutes)'], 
                         df_weather.Humidity.max()+0.5, df_weather.Humidity.min()-0.5, 
                         facecolor="blue", color="blue", alpha=0.1, zorder=0, label='Rain')
axes[0,1].legend(loc='upper right')
axes[0,1].set_title('Track Humidity (%)')

# Pressure
sns.lineplot(data=df_weather, x='SessionTime(Minutes)', y='Pressure', ax=axes[1,0])
axes[1,0].set_title('Air Pressure (mbar)')

# Wind Direction & Speed
rect = axes[1,1].get_position()
wax = WindroseAxes(fig, rect)
fig.add_axes(wax)
wax.bar(df_weather.WindDirection, df_weather.WindSpeed, normed=True, opening=0.8, edgecolor='white')
wax.set_legend()
axes[1,1].set_title('Wind Direction (°) and Speed(m/s)')

plt.show()

# Print top 3 drivers for verification
print(f"Top 3 {session} Results:")
print(df_results[['Position', 'Abbreviation', 'TimeDiff']].head(3))