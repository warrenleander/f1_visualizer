# Formula 1 Race Analysis Script

This Python script analyzes Formula 1 race data using the FastF1 library. It generates various visualizations and statistics for a specified F1 session.

## Features

- Track performance analysis for top 3 drivers
- Session results visualization
- Sector time analysis (for qualifying and practice sessions)
- Team speed analysis
- Weather data visualization (temperature, humidity, pressure, wind)

## Prerequisites

Before running this script, ensure you have the following libraries installed:

- fastf1
- matplotlib
- numpy
- pandas
- seaborn
- windrose

## Usage

1. Set the following variables at the beginning of the script:
   - `year`: The year of the race (e.g., 2023)
   - `location`: The location/name of the race (e.g., 'Monza')
   - `session`: The session type ('Q' for qualifying, 'R' for race, 'FP1', 'FP2', or 'FP3' for practice sessions)

2. Run the script:
python formula_one_analysis.py

3. The script will generate several plots and print the top 3 results for the specified session.

## Outputs

The script generates the following visualizations:

1. Track Performance plot: Shows the performance of the top 3 drivers around the track.
2. Results plot: Displays the gap to the first place (for races) or fastest time (for other sessions).
3. Sector plots (for non-race sessions): Shows the top 10 times for each sector.
4. Weather plots: Visualizes track and air temperature, humidity, air pressure, and wind data.

## Notes

- The script uses FastF1's caching system to speed up subsequent runs. The cache is stored in a 'fastf1_cache' directory in the same location as the script.
- Make sure you have an active internet connection when running the script for the first time to download the required data.

## Customization

You can modify the script to analyze different races or sessions by changing the `year`, `location`, and `session` variables at the beginning of the script.
