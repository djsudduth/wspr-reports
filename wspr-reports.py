import io
import sys
import argparse
import statistics
import numpy as np
import pandas as pd
from bisect import bisect
import matplotlib.pyplot as plt
import matplotlib
import math
import random

NEAR = 800
MID = 3000
LONG = 6000
VLONG = 13000

# Use a non-interactive backend for matplotlib if running in an environment without a display
try:
    matplotlib.use('Agg')
except ImportError:
    print("matplotlib.use('Agg') failed. Plotting might require a GUI backend.")


WSPR_HEADER_ERR = "Your wspr file cannot be missing or have spaces in the data headers that defaults from wsprnet.org"

def open_wspr_file():
    #open the wspr data into a dataframe
    # The WSPR data file is expected to be named "wspr.txt" and be in the same directory
    df = pd.read_csv("wspr.txt", sep='\t')
    return (df)

def open_goes_xray_file(xray_download):
    #open the GOES satellite 6-hour xray flux data
    # The GOES data file is expected to be named "xrays-6-hour.json"
    goes_xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"

    if (xray_download):
        print(f"\n--- Downloading GOES X-ray data from: {goes_xray_url} ---")
        dfx = pd.read_json(goes_xray_url)
        dfx.to_json("xrays-6-hour.json", orient='records', indent=2, date_format='iso')
    else:
        dfx = pd.read_json('xrays-6-hour.json')
    # Select every other row as per original script logic
    dfx = dfx.iloc[::2]
    dfx = dfx.rename(columns={'time_tag':'Timestamp'})
    dfx.flux = dfx.flux.apply(float).round(12)
    dfx['modflux'] = dfx['flux'] * 1e8
    # Convert 'Timestamp' to datetime objects
    dfx['Timestamp'] = pd.to_datetime(dfx['Timestamp'])
    return(dfx)


def join_wspr_with_goes(df, dfx):
    #join the GOES satellite 6-hour xray flux data with the wspr data and return join
    df = pd.merge(df, dfx, on='Timestamp', how='inner').reset_index()
    df.to_csv("wspr-goes-data.csv")
    return(df)


def add_wspr_dimensions(df):
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].str.strip()
        df['Time'] = df['Timestamp'].str[-5:]
        # Ensure Timestamp is in the correct format for merging (YYYY-MM-DDTHH:MM:SSZ)
        # Convert to datetime objects here
        df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.replace(" ", "T") +":00Z", errors='coerce')
        # Drop rows where Timestamp conversion failed
        df.dropna(subset=['Timestamp'], inplace=True)

        bins_az = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
        labels_az = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        df['az'] = pd.to_numeric(df['az'], errors='coerce')

        df['map'] = pd.cut(df['az'],
                           bins=bins_az,
                           labels=labels_az,
                           ordered=False, include_lowest=True)

        df['km'] = pd.to_numeric(df['km'], errors='coerce')
        df['drange'] = pd.cut(df['km'],
                              bins=[0, 800, 3000, 6000, 13000, np.inf],
                              labels=['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG'],
                              ordered=False)

        df = df.sort_values('Timestamp').reset_index(drop=True)
        print ("\n\nWSPR and GOES joined data - first rows: ")
        print (df.head(7) )
        return(df)
    else:
        print ("\n" + WSPR_HEADER_ERR)
        return None


def get_wspr_snr_trends(df):
    if df is None or df.empty:
        print("No data to process for SNR trends.")
        return

    df['SNR'] = pd.to_numeric(df['SNR'], errors='coerce')
    df.dropna(subset=['SNR', 'map', 'km', 'Reporter'], inplace=True)

    if df.empty:
        print("Data became empty after dropping NaNs for SNR trends analysis.")
        return

    df2 = df.groupby(['map', 'km', 'Reporter'], observed=True)['SNR'].describe()
    print ("\n\nWSPR mean and std dev of SNRs by map direction from your callsign location: ")
    print (df2.to_string() + "\n\n")
    df2.to_csv("wspr-map-trends.csv")


def create_polar_plot(df, output_filename="wspr_polar_plot.png"):
    if df is None or df.empty:
        print("No data to create plot.")
        return

    df['SNR'] = pd.to_numeric(df['SNR'], errors='coerce')
    df.dropna(subset=['SNR', 'map', 'drange', 'Reporter'], inplace=True)

    if df.empty:
        print("Data is empty after processing for plotting (after initial dropna).")
        return

    bar_start_baseline_snr = -25.0
    radial_axis_padding = 3.0

    direction_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    drange_definitions = {
        'NEAR': f'(0-{NEAR} km)',
        'MID': f'({NEAR}-{MID} km)',
        'LONG': f'({MID}-{LONG} km)',
        'VLONG': f'({LONG}-{VLONG} km)',
        'VVLONG': f'({VLONG}+ km)'
    }    
    drange_order = ['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG']
    df['drange'] = pd.Categorical(df['drange'], categories=drange_order, ordered=True)
    df['map'] = pd.Categorical(df['map'], categories=direction_order, ordered=True)

    # Define num_dranges
    num_dranges = len(drange_order)

    # Calculate descriptive statistics for each Reporter within each map/drange
    reporter_snr_stats = df.groupby(['map', 'drange', 'Reporter'], observed=False)['SNR'].describe()

    # Extract the 'mean' for the primary plot
    raw_mean_values_df = df.groupby(['map', 'drange'], observed=False)['SNR'].mean().unstack()
    raw_mean_values_df = raw_mean_values_df.reindex(index=direction_order).reindex(columns=drange_order)

    # Calculate the average of the 'std' values from the reporter_snr_stats
    averaged_std_values_df = reporter_snr_stats['std'].reset_index()
    averaged_std_values_df = averaged_std_values_df.groupby(['map', 'drange'], observed=False)['std'].mean().unstack()
    averaged_std_values_df = averaged_std_values_df.reindex(index=direction_order).reindex(columns=drange_order)
    plot_std_values_df = averaged_std_values_df

    # Calculate total observations (counts) for each map/drange bin
    counts_df = df.groupby(['map', 'drange'], observed=False).size().unstack()
    counts_df = counts_df.reindex(index=direction_order).reindex(columns=drange_order)

    effective_std_for_ylim = plot_std_values_df.fillna(0)

    num_directions = len(direction_order)
    angles_radians = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

    min_data_points_for_ylim = raw_mean_values_df - effective_std_for_ylim
    max_data_points_for_ylim = raw_mean_values_df + effective_std_for_ylim

    true_min_data_extent = min_data_points_for_ylim.min().min() if not min_data_points_for_ylim.empty else np.nan
    true_max_data_extent = max_data_points_for_ylim.max().max() if not max_data_points_for_ylim.empty else np.nan

    if pd.isna(true_min_data_extent):
        min_val_for_scaling = raw_mean_values_df.min().min() if not raw_mean_values_df.empty else np.nan
        if pd.isna(min_val_for_scaling): min_val_for_scaling = bar_start_baseline_snr
    else:
        min_val_for_scaling = true_min_data_extent

    if pd.isna(true_max_data_extent):
        max_val_for_scaling = raw_mean_values_df.max().max() if not raw_mean_values_df.empty else np.nan
        if pd.isna(max_val_for_scaling): max_val_for_scaling = 0.0
    else:
        max_val_for_scaling = true_max_data_extent

    plot_min_r = min(min_val_for_scaling, bar_start_baseline_snr) - radial_axis_padding
    target_outer_limit = max_val_for_scaling + 5.0
    plot_max_r = 5.0 * math.floor(target_outer_limit / 5.0)

    if plot_max_r <= plot_min_r:
        plot_max_r = plot_min_r + 10.0

    ax.set_ylim(plot_min_r, plot_max_r)

    current_plot_radial_span = plot_max_r - plot_min_r
    if current_plot_radial_span <= 0:
        current_plot_radial_span = 10
    text_offset_magnitude_for_labels = current_plot_radial_span * 0.02

    num_directions = len(direction_order)
    total_sector_width = (2 * np.pi / num_directions)
    slot_width_for_each_drange = total_sector_width / (num_dranges + 1.5)
    std_dev_range_bar_visual_width = slot_width_for_each_drange * 0.95
    mean_bar_visual_width = std_dev_range_bar_visual_width * 0.6

    # Define a small angular offset for the count label
    # This value might need slight tuning based on your data density and desired look
    count_label_angular_offset = slot_width_for_each_drange * 0.25

    for i, drange_cat in enumerate(drange_order):
        if drange_cat in raw_mean_values_df.columns:
            mean_snr_series = raw_mean_values_df.loc[:, drange_cat].reindex(direction_order)
            plot_snr_means = mean_snr_series.fillna(bar_start_baseline_snr).values

            std_dev_series = plot_std_values_df.loc[:, drange_cat].reindex(direction_order)
            plot_snr_stds = std_dev_series.fillna(0).values

            current_count_series = counts_df.loc[:, drange_cat].reindex(direction_order)
            plot_counts = current_count_series.fillna(0).values

            angle_offset = (i - (num_dranges - 1) / 2.0) * slot_width_for_each_drange
            current_bar_angles = angles_radians + angle_offset

            km_range_text = drange_definitions.get(drange_cat, "")
            full_legend_label = f"{drange_cat} {km_range_text}".strip()

            std_range_bar_bottom_values = plot_snr_means - plot_snr_stds
            std_range_bar_height_values = 2 * plot_snr_stds
            std_range_color = 'lightgray'
            ax.bar(current_bar_angles, std_range_bar_height_values,
                   width=std_dev_range_bar_visual_width,
                   alpha=0.5,
                   bottom=std_range_bar_bottom_values,
                   color=std_range_color,
                   zorder=2,
                   edgecolor='darkgray',
                   linewidth=0.5)

            mean_bar_heights = plot_snr_means - bar_start_baseline_snr
            mean_bars_patches = ax.bar(current_bar_angles, mean_bar_heights,
                               width=mean_bar_visual_width,
                               label=full_legend_label,
                               alpha=0.85,
                               bottom=bar_start_baseline_snr,
                               zorder=3)

            # Iterate for text labels
            for idx in range(len(plot_snr_means)):
                mean_val = plot_snr_means[idx]
                std_val = plot_snr_stds[idx]
                angle_val = current_bar_angles[idx]
                count_val = plot_counts[idx]

                # Calculate bounds, even if std_val is 0, for positioning the count label
                lower_bound = mean_val - std_val
                upper_bound = mean_val + std_val

                # --- Mean Label ---
                mean_bar_length = mean_val - bar_start_baseline_snr
                if abs(mean_bar_length) > 0.01:
                    mean_label_offset_sign = np.sign(mean_bar_length)
                    if mean_label_offset_sign == 0: mean_label_offset_sign = 1

                    text_pos_mean = mean_val + (mean_label_offset_sign * text_offset_magnitude_for_labels)
                    ax.text(angle_val, text_pos_mean, f'{mean_val:.1f}',
                            ha='center', va='bottom' if mean_label_offset_sign >= 0 else 'top',
                            fontsize=7, color='black', zorder=4)

                # --- Std Dev Bound Labels ---
                if std_val > 0.01:
                    std_label_font_size = 6
                    std_label_color = 'darkslateblue'

                    # Upper Bound Label (mean + std)
                    text_pos_upper = upper_bound + text_offset_magnitude_for_labels
                    ax.text(angle_val, text_pos_upper, f'{upper_bound:.1f}',
                            ha='center', va='bottom',
                            fontsize=std_label_font_size, color=std_label_color, zorder=4)

                    # Lower Bound Label (mean - std)
                    text_pos_lower = lower_bound - text_offset_magnitude_for_labels
                    ax.text(angle_val, text_pos_lower, f'{lower_bound:.1f}',
                            ha='center', va='top',
                            fontsize=std_label_font_size, color=std_label_color, zorder=4)

                # --- Adjusted: Total Observations Label ---
                if count_val > 0:
                    # Shift the angle slightly to the right of the mean/std labels
                    adjusted_angle_for_count = angle_val + count_label_angular_offset + 0.067
                    # Position it relative to the upper bound, but with horizontal offset
                    text_pos_count = upper_bound + (text_offset_magnitude_for_labels * 0.7)
                    ax.text(adjusted_angle_for_count, text_pos_count, f'(N={int(count_val)})',
                            ha='left', va='bottom',
                            fontsize=6, color='darkgreen', zorder=4)

    ax.set_xticks(angles_radians)
    ax.set_xticklabels(direction_order, fontsize=10)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    shifted_grid_degrees = np.array([337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5])
    shifted_grid_radians = np.deg2rad(shifted_grid_degrees)
    shifted_grid_radians[shifted_grid_radians < 0] += 2 * np.pi
    for angle_rad in shifted_grid_radians:
        ax.plot([angle_rad, angle_rad], [ax.get_ylim()[0], ax.get_ylim()[1]], color='gray', linestyle='-', linewidth=1.4)

    if plot_min_r <= 0 <= plot_max_r:
        ax.plot(np.append(angles_radians, angles_radians[0]), [0]*(num_directions+1), color='grey', linestyle='--', linewidth=0.3)

    if plot_min_r <= bar_start_baseline_snr <= plot_max_r:
         ax.plot(np.append(angles_radians, angles_radians[0]), [bar_start_baseline_snr]*(num_directions+1), color='blue', linestyle=':', linewidth=0.6, alpha=0.6)

    ax.set_title("Average WSPR SNR by Direction and Distance Range (±Avg of Per-Reporter Std Dev)", va='bottom', fontsize=16, pad=25)
    ax.legend(title="Distance Range (km)", loc="upper right", bbox_to_anchor=(1.2, 1.05), fontsize=9, title_fontsize=10)

    fig.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"\nPolar plot saved as {output_filename}")


def create_time_series_plot(df, dfx_original, output_filename="wspr_xray_timeseries.png"):
    """
    Plots a time series of SNR for up to 3 random reporters within the same
    map and drange, along with the GOES modflux.
    """
    df_plot = df.copy()
    # Timestamp should already be datetime from add_wspr_dimensions, but ensure here
    df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'], errors='coerce')

    # Drop rows where essential plotting columns are NaN for WSPR data
    df_plot.dropna(subset=['SNR', 'map', 'drange', 'Reporter', 'Timestamp'], inplace=True)

    if df_plot.empty:
        print("No valid data after filtering for time series plot (WSPR data).")
        # Still proceed to plot GOES flux if WSPR data is empty
        pass


    # Find combinations of map and drange with at least 3 unique reporters and sufficient data
    suitable_groups = df_plot.groupby(['map', 'drange'], observed=False).agg(
        unique_reporters=('Reporter', 'nunique'),
        data_points=('Timestamp', 'size')
    ).reset_index()

    suitable_groups = suitable_groups[
        (suitable_groups['unique_reporters'] >= 3) &
        (suitable_groups['data_points'] > 40)
    ]

    selected_map = None
    selected_drange = None
    if not suitable_groups.empty:
        chosen_group = suitable_groups.sample(n=1, random_state=42).iloc[0]
        selected_map = 'SE' #chosen_group['map']
        selected_drange = 'MID' #chosen_group['drange']
        print(f"Selected map: {selected_map}, drange: {selected_drange} for time series plot.")
    else:
        print("Could not find a 'map' and 'drange' combination with at least 3 unique reporters and sufficient WSPR data. Plotting GOES flux only.")

    df_filtered_snr = pd.DataFrame()
    selected_reporters = []
    earliest_reporter_timestamp = None

    if selected_map and selected_drange:
        df_filtered_snr = df_plot[(df_plot['map'] == selected_map) & (df_plot['drange'] == selected_drange)].copy()

        if not df_filtered_snr.empty:
            unique_reporters_in_group = df_filtered_snr['Reporter'].unique().tolist()
            num_reporters_to_plot = min(3, len(unique_reporters_in_group))
            if num_reporters_to_plot > 0:
                selected_reporters = random.sample(unique_reporters_in_group, num_reporters_to_plot)
                print(f"Selected reporters: {selected_reporters}")
                df_filtered_snr = df_filtered_snr[df_filtered_snr['Reporter'].isin(selected_reporters)].copy()
                df_filtered_snr['minute'] = df_filtered_snr['Timestamp'].dt.minute
                df_filtered_snr = df_filtered_snr[df_filtered_snr['minute'] % 1 == 0].copy()
                df_filtered_snr.sort_values(by='Timestamp', inplace=True)
                if not df_filtered_snr.empty:
                    earliest_reporter_timestamp = df_filtered_snr['Timestamp'].min()
            else:
                print("Not enough unique reporters in selected group for SNR plotting.")
                df_filtered_snr = pd.DataFrame() # Clear df_filtered_snr if no reporters to plot
        else:
            print(f"No WSPR data for selected map '{selected_map}' and drange '{selected_drange}'.")

    # Fallback for earliest_reporter_timestamp if no suitable group or reporters were found
    if earliest_reporter_timestamp is None and not df_plot.empty:
        earliest_reporter_timestamp = df_plot['Timestamp'].min()
    elif earliest_reporter_timestamp is None: # If df_plot is also empty
        # Default to GOES data start if no WSPR data at all
        earliest_reporter_timestamp = dfx_original['Timestamp'].min()

    # Calculate GOES start time (12 minutes before earliest reporter data)
    goes_start_time = earliest_reporter_timestamp - pd.Timedelta(minutes=12)

    # Prepare GOES modflux data (independent of WSPR data presence)
    dfx_plot = dfx_original.copy()
    dfx_plot['Timestamp'] = pd.to_datetime(dfx_plot['Timestamp'])
    dfx_plot.dropna(subset=['Timestamp', 'modflux'], inplace=True)
    dfx_plot['minute'] = dfx_plot['Timestamp'].dt.minute
    modflux_data = dfx_plot[dfx_plot['minute'] % 4 == 0].copy()
    modflux_data = modflux_data[modflux_data['Timestamp'] >= goes_start_time].copy() # Filter by start time
    modflux_data = modflux_data.groupby('Timestamp')['modflux'].mean().reset_index()
    modflux_data.sort_values(by='Timestamp', inplace=True)


    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot SNR for each selected reporter (if data exists)
    if not df_filtered_snr.empty:
        for reporter in selected_reporters:
            reporter_df = df_filtered_snr[df_filtered_snr['Reporter'] == reporter]
            if not reporter_df.empty:
                ax1.plot(reporter_df['Timestamp'], reporter_df['SNR'], label=f'SNR ({reporter})')
        ax1.set_ylabel('WSPR SNR (dB)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        if selected_map and selected_drange:
            ax1.set_title(f'WSPR SNR vs. GOES X-ray Flux for {selected_map} {selected_drange}\nSelected Reporters: {", ".join(selected_reporters)}')
        else:
            ax1.set_title(f'WSPR SNR vs. GOES X-ray Flux\n(No suitable WSPR group found, showing available SNR data and GOES Flux)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
    else:
        # If no SNR data to plot, set a placeholder title and only plot GOES
        ax1.set_title(f'GOES X-ray Flux (No WSPR SNR data to plot)')
        # Hide y-axis ticks and labels for SNR if no data
        ax1.set_yticks([])
        ax1.set_yticklabels([])

    ax1.set_xlabel('Time (UTC)')

    # Create a second y-axis for modflux
    ax2 = ax1.twinx()
    ax2.plot(modflux_data['Timestamp'], modflux_data['modflux'],
             color='red', linestyle='--', label='Average GOES modflux')
    ax2.set_ylabel('GOES modflux (x$10^{-8}$ W/m$^2$)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Time series plot saved as {output_filename}")



def create_directional_avg_slope_line_plot(df_reporter_trends, output_filename="wspr_avg_slope_by_direction_dist_line_plot.png"):
    """
    Creates a line plot showing the average SNR slope for each cardinal direction,
    with separate lines for each distance range.

    Args:
        df_reporter_trends (pd.DataFrame): DataFrame containing 'slope', 'map', and 'km'
                                           columns for individual reporter trends.
        output_filename (str): Filename for the saved plot.
    """
    if df_reporter_trends is None or df_reporter_trends.empty:
        print("No reporter trend data to create average slope by direction & distance line plot.")
        return

    if not all(col in df_reporter_trends.columns for col in ['slope', 'map', 'km']):
        print("Error: DataFrame for line plot is missing 'slope', 'map', or 'km' column.")
        return

    df_plot = df_reporter_trends.copy()

    df_plot['slope'] = pd.to_numeric(df_plot['slope'], errors='coerce')
    df_plot.dropna(subset=['slope', 'map', 'km'], inplace=True)

    if df_plot.empty:
        print("No valid data remaining after cleaning for average slope by direction & distance line plot.")
        return

    direction_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    bins_km = [0, 800, 4000, 8000, 13000, np.inf]
    drange_order = ['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG'] # Consistent order for lines/legend
    drange_legend_definitions = {
        'NEAR': '(0-800 km)', 'MID': '(800-4000 km)', 
        'LONG': '(4000-8000 km)', 'VLONG': '(8000-13000 km)', 
        'VVLONG': '(13000+ km)'
    }

    df_plot['map'] = pd.Categorical(df_plot['map'], categories=direction_order, ordered=True)
    df_plot['drange'] = pd.cut(df_plot['km'], bins=bins_km, labels=drange_order, right=True, include_lowest=True)
    df_plot.dropna(subset=['drange'], inplace=True) # Remove rows if km didn't fall into a bin
    df_plot['drange'] = pd.Categorical(df_plot['drange'], categories=drange_order, ordered=True)


    # Calculate the average slope for each direction and distance combination
    avg_slopes_by_dir_dist = df_plot.groupby(['map', 'drange'], observed=False)['slope'].mean()
    
    # Pivot the data: 'map' as index, 'drange' as columns, values are avg slopes
    avg_slopes_pivot = avg_slopes_by_dir_dist.unstack(level='drange')
    
    # Reindex to ensure all directions and dranges are present and in order for plotting
    avg_slopes_pivot = avg_slopes_pivot.reindex(index=direction_order)
    avg_slopes_pivot = avg_slopes_pivot.reindex(columns=drange_order) # Ensures consistent column order

    if avg_slopes_pivot.isnull().all().all():
        print("No data to plot after pivoting for average slopes by direction and distance.")
        return

    plt.figure(figsize=(12, 7)) # Slightly wider for potentially more legend items
    ax = plt.gca()

    # Plot each distance range as a separate line
    for drange_cat in drange_order: # Iterate in defined order for consistent legend/colors
        if drange_cat in avg_slopes_pivot.columns:
            series_to_plot = avg_slopes_pivot[drange_cat]
            
            # Only plot if there's some non-NaN data for this distance range
            if not series_to_plot.isnull().all():
                km_range_text = drange_legend_definitions.get(drange_cat, "")
                full_legend_label = f"{drange_cat} {km_range_text}".strip()
                ax.plot(series_to_plot.index, series_to_plot.values, 
                        marker='o', linestyle='-', label=full_legend_label)

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=1)

    ax.set_title("Average Trend of SNR Slopes by Direction and Distance", fontsize=16)
    ax.set_xlabel("Direction", fontsize=12)
    ax.set_ylabel("Average SNR Slope (dB / report index)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.set_xticks(range(len(direction_order)))
    ax.set_xticklabels(direction_order)

    # Adjust legend position
    ax.legend(title="Distance Range (km)", loc='upper left', bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust rect to make space for legend
    plt.savefig(output_filename)
    plt.close()
    print(f"\nAverage slope by direction and distance line plot saved as {output_filename}")




def calculate_reporter_snr_trends(df_with_map_and_km):
    if df_with_map_and_km is None or df_with_map_and_km.empty:
        print("Input DataFrame for calculating reporter SNR trends is empty.")
        return None
    required_cols = ['km', 'Reporter', 'SNR', 'map']
    if not all(col in df_with_map_and_km.columns for col in required_cols):
        print(f"Input DataFrame for trend calculation is missing required columns: {required_cols}")
        return None

    df_calc = df_with_map_and_km.copy()
    df_calc['SNR'] = pd.to_numeric(df_calc['SNR'], errors='coerce')
    df_calc.dropna(subset=['km', 'Reporter', 'SNR', 'map'], inplace=True)
    if df_calc.empty:
        print("DataFrame is empty after coercing SNR and dropping NaNs for trend calculation.")
        return None

    try:
        df_trends_intermediate = df_calc.groupby(['km', 'Reporter']).agg(
            snr_values_list=('SNR', list),
            map=('map', 'first')
        ).reset_index()
    except Exception as e:
        print(f"Error during initial grouping for trend calculation: {e}")
        return None
    if df_trends_intermediate.empty:
        print("No data after grouping by km and Reporter for trend calculation.")
        return None

    slopes = []
    snr_stdvs_per_reporter = []
    num_reports_list = []

    for index, row in df_trends_intermediate.iterrows():
        snr_list_for_reporter = row['snr_values_list']
        num_reports = len(snr_list_for_reporter)
        num_reports_list.append(num_reports)

        if num_reports > 2:
            s_indices = list(range(num_reports))
            try:
                slope_val, _ = statistics.linear_regression(s_indices, snr_list_for_reporter)
                stdv_val = 0.0
                if len(set(snr_list_for_reporter)) > 1: # stdev requires at least two different data points
                    stdv_val = statistics.stdev(snr_list_for_reporter)
                slopes.append(round(float(slope_val), 2))
                snr_stdvs_per_reporter.append(round(float(stdv_val), 2))
            except statistics.StatisticsError:
                slopes.append(0.0) 
                snr_stdvs_per_reporter.append(0.0)
            except Exception as e_stat:
                print(f"Unexpected error in statistics for {row['Reporter']} SNRs {snr_list_for_reporter}: {e_stat}")
                slopes.append(np.nan)
                snr_stdvs_per_reporter.append(np.nan)
        else:
            slopes.append(np.nan)
            snr_stdvs_per_reporter.append(np.nan)

    df_trends_intermediate['slope'] = slopes
    df_trends_intermediate['snr_stdv_per_reporter'] = pd.to_numeric(snr_stdvs_per_reporter, errors='coerce')
    df_trends_intermediate['num_reports'] = num_reports_list
    
    df_final_trends = df_trends_intermediate.dropna(subset=['slope', 'map']).copy()
    if df_final_trends.empty:
        print("No valid reporter trends calculated (all slopes were NaN or map was missing).")
        return None
        
    print("\n\nCalculated Reporter SNR Trends (first 5 rows):")
    print(df_final_trends.head(5).to_string(index=False))
    df_final_trends.to_csv("wspr-reporter-trends-calculated.csv", index=False)
    print("Saved calculated reporter trends to wspr-reporter-trends-calculated.csv")
    return df_final_trends


def main():

    parser = argparse.ArgumentParser(
        description="Process WSPR data, optionally download live GOES X-ray data to file, and generate plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    parser.add_argument(
        '-x', '--download-xray',
        action='store_true', 
        help="Download live GOES X-ray data. If not specified, the manually downloaded file is used. (default: False)"
    )

    # parser.add_argument('--output-dir', type=str, default='.', help='Directory to save output files.')

    args = parser.parse_args()

    try:
        xray_download = args.download_xray

        df = open_wspr_file()
        df = add_wspr_dimensions(df)
        dfx = open_goes_xray_file(xray_download) # Load dfx once here

        if df is not None:
            df_joined = join_wspr_with_goes(df, dfx) # Join with original dfx
            get_wspr_snr_trends(df_joined) # Use joined df
            create_polar_plot(df_joined, "wspr_snr_polar_plot.png") # Use joined df
            create_time_series_plot(df_joined, dfx, "wspr_xray_timeseries_plot.png") # Pass both joined df and original dfx
            df_reporter_trends = calculate_reporter_snr_trends(df_joined.copy())
            create_directional_avg_slope_line_plot(df_reporter_trends, "wspr_avg_slope_by_direction_line_plot.png")
        else:
            print("Failed to process WSPR dimensions. Aborting further analysis.")

    except FileNotFoundError as fnf_error:
        print(f"Error: File not found. Please ensure '{fnf_error.filename}' is in the correct location.")
    except pd.errors.EmptyDataError as ede_error:
        print(f"Error: One of the data files is empty. ({ede_error})")
    except KeyError as ke_error:
        print(f"Error: A required column is missing from the data: {ke_error}. Check file headers and content.")
    except Exception as e:
        print("Could not execute wspr-reports: ", repr(e))

if __name__ == '__main__':
    main()
    

# Possible APIs
# https://www.wsprnet.org/olddb?mode=html&band=20&limit=1000&findcall=&findreporter=&sort=date
# http://db1.wspr.live/?query=SELECT%20*%20FROM%20wspr.rx%20WHERE%20tx_sign%3D%27KN0VA%27%20LIMIT%2010