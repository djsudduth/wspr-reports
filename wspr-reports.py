import io
import statistics
import numpy as np
import pandas as pd
from bisect import bisect
import matplotlib.pyplot as plt
import matplotlib
import math

# Use a non-interactive backend for matplotlib if running in an environment without a display
try:
    matplotlib.use('Agg')
except ImportError:
    print("matplotlib.use('Agg') failed. Plotting might require a GUI backend.")


WSPR_HEADER_ERR = "Your wspr file cannot be missing or have spaces in the data headers that defaults from wsprnet.org"

## -----------------------------------------------------------------------------
## Data Loading and Initial Processing Functions
## -----------------------------------------------------------------------------

def open_wspr_file():
    #open the wspr data into a dataframe
    # The WSPR data file is expected to be named "wspr.txt" and be in the same directory
    try:
        df = pd.read_csv("wspr.txt", sep='\t')
        if df.empty:
            print("Warning: wspr.txt is empty.")
        return df
    except FileNotFoundError:
        print("Error: wspr.txt not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: wspr.txt is empty (pandas EmptyDataError).")
        return None
    except Exception as e:
        print(f"Error opening wspr.txt: {e}")
        return None

def open_goes_xray_file():
    #open the GOES satellite 6-hour xray flux data
    # The GOES data file is expected to be named "xrays-6-hour.json"
    try:
        dfx = pd.read_json('xrays-6-hour.json')
        if dfx.empty:
            print("Warning: xrays-6-hour.json is empty.")
            return None # Return None if empty to handle it in main
    except FileNotFoundError:
        print("Error: xrays-6-hour.json not found.")
        return None
    except ValueError as ve: # Handles issues with JSON parsing
        print(f"Error parsing xrays-6-hour.json: {ve}")
        return None
    except Exception as e:
        print(f"Error opening xrays-6-hour.json: {e}")
        return None

    # Select every other row as per original script logic
    dfx = dfx.iloc[::2].copy() # Use .copy() to avoid SettingWithCopyWarning
    dfx.rename(columns={'time_tag':'Timestamp'}, inplace=True)
    dfx['flux'] = pd.to_numeric(dfx['flux'], errors='coerce').round(12) # Coerce ensures non-numeric become NaN
    dfx.dropna(subset=['flux'], inplace=True) # Drop rows where flux could not be parsed
    dfx['modflux'] = dfx['flux'] * 1e8
    return dfx


def join_wspr_with_goes(df_wspr, df_goes):
    if df_wspr is None or df_goes is None:
        print("Cannot join dataframes; one or both are missing.")
        return None
    #join the GOES satellite 6-hour xray flux data with the wspr data and return join
    df_merged = pd.merge(df_wspr, df_goes, on='Timestamp', how='inner').reset_index(drop=True)
    if df_merged.empty:
        print("Warning: Merged WSPR and GOES data is empty. Check Timestamp formats and data overlap.")
    else:
        df_merged.to_csv("wspr-goes-data.csv", index=False)
        print("Saved merged WSPR and GOES data to wspr-goes-data.csv")
    return df_merged


def add_wspr_dimensions(df_input):
    if df_input is None or df_input.empty:
        print("Input DataFrame to add_wspr_dimensions is empty or None.")
        return None
    
    df = df_input.copy()

    if 'Timestamp' not in df.columns:
        print(f"Error: Timestamp column missing. Available columns: {df.columns.tolist()}")
        return None
    if 'Reporter' not in df.columns:
        print(f"Error: Reporter column missing. Available columns: {df.columns.tolist()}")
        return None
    if 'az' not in df.columns:
        print(f"Error: az column missing. Available columns: {df.columns.tolist()}")
        return None
    if 'km' not in df.columns:
        print(f"Error: km column missing. Available columns: {df.columns.tolist()}")
        return None

    df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
    df['Time'] = df['Timestamp'].str[-5:] # Assumes format like YYYY-MM-DD HH:MM
    # Ensure Timestamp is in the correct format for merging (YYYY-MM-DDTHH:MM:SSZ like)
    # This was specific to GOES data format. If WSPR Timestamp is just YYYY-MM-DD HH:MM,
    # and GOES is YYYY-MM-DDTHH:MM:SSZ, adjust accordingly or ensure pre-processing.
    # Assuming WSPR 'Timestamp' from wspr.txt is 'YYYY-MM-DD HH:MM'
    # And GOES 'Timestamp' after rename is 'YYYY-MM-DDTHH:MM:SSZ'
    # The join_wspr_with_goes needs compatible formats.
    # For now, let's assume the Timestamp formatting for the join is handled or compatible.
    # The example for Timestamp transformation seems to be for WSPR data to match GOES:
    if not df['Timestamp'].str.contains("T").any(): # If not already in ISO-like format
        df['Timestamp'] = df['Timestamp'].str.replace(" ", "T", n=1) + ":00Z" # Example transformation

    df['Reporter'] = df['Reporter'].astype(str).str.strip()

    bins_az = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    labels_az = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'] # Last 'N' handles 360 degrees
    df['az'] = pd.to_numeric(df['az'], errors='coerce')
    df.dropna(subset=['az'], inplace=True) # Remove rows where azimuth is not valid

    df['map'] = pd.cut(df['az'],
                       bins=bins_az,
                       labels=labels_az[:-1], # Use one less label than bins if last bin edge is inclusive for N
                       ordered=False, include_lowest=True, right=True) # Ensure 0 and 360 are handled.
    # Handle cases that might fall on 360 exactly if last label was N
    if 360.0 in bins_az:
      df.loc[df['az'] == 360.0, 'map'] = 'N'


    df['km'] = pd.to_numeric(df['km'], errors='coerce')
    df.dropna(subset=['km'], inplace=True) # Remove rows where km is not valid

    df['drange'] = pd.cut(df['km'],
                          bins=[0, 800, 4000, 8000, 13000, np.inf],
                          labels=['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG'],
                          ordered=False, include_lowest=True, right=True)
    df.dropna(subset=['drange'], inplace=True)


    df = df.sort_values('Timestamp').reset_index(drop=True)
    print ("\n\nWSPR data with dimensions added (first 5 rows): ")
    print (df.head(5).to_string())
    return df

## -----------------------------------------------------------------------------
## Plot 1: Average SNR Polar Plot (Mean SNR with its Std Dev)
## -----------------------------------------------------------------------------

def create_polar_plot(df_input, output_filename="wspr_avg_snr_polar_plot.png"):
    if df_input is None or df_input.empty:
        print("No data to create average SNR plot.")
        return

    df = df_input.copy()
    df['SNR'] = pd.to_numeric(df['SNR'], errors='coerce')
    df.dropna(subset=['SNR', 'map', 'drange'], inplace=True)

    if df.empty:
        print("Data is empty for average SNR plot (after initial dropna).")
        return

    bar_start_baseline_snr = -25.0
    radial_axis_padding = 3.0

    direction_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    drange_definitions = {
        'NEAR': '(0-800 km)', 'MID': '(800-4000 km)',
        'LONG': '(4000-8000 km)', 'VLONG': '(8000-13000 km)',
        'VVLONG': '(13000+ km)'
    }
    drange_order = ['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG']
    df['drange'] = pd.Categorical(df['drange'], categories=drange_order, ordered=True)
    df['map'] = pd.Categorical(df['map'], categories=direction_order, ordered=True)

    # Calculate Mean and Std Dev of SNRs for each group
    agg_funcs = {'SNR': ['mean', 'std']}
    grouped_data = df.groupby(['map', 'drange'], observed=False).agg(agg_funcs)
    grouped_data.columns = ['SNR_mean', 'SNR_std'] # Flatten MultiIndex columns
    grouped_data = grouped_data.reset_index()

    raw_mean_values_df = grouped_data.pivot_table(index='map', columns='drange', values='SNR_mean')
    raw_mean_values_df = raw_mean_values_df.reindex(index=direction_order).reindex(columns=drange_order)

    raw_std_values_df = grouped_data.pivot_table(index='map', columns='drange', values='SNR_std')
    raw_std_values_df = raw_std_values_df.reindex(index=direction_order).reindex(columns=drange_order)
    effective_std_for_ylim = raw_std_values_df.fillna(0)

    num_directions = len(direction_order)
    angles_radians = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

    min_data_points_for_ylim = raw_mean_values_df - effective_std_for_ylim
    max_data_points_for_ylim = raw_mean_values_df + effective_std_for_ylim
    true_min_data_extent = min_data_points_for_ylim.min().min(skipna=True)
    true_max_data_extent = max_data_points_for_ylim.max().max(skipna=True)

    min_val_for_scaling = bar_start_baseline_snr
    if pd.notna(true_min_data_extent): min_val_for_scaling = true_min_data_extent
    
    max_val_for_scaling = 0.0 
    if pd.notna(true_max_data_extent): max_val_for_scaling = true_max_data_extent
    
    plot_min_r = min(min_val_for_scaling, bar_start_baseline_snr) - radial_axis_padding
    target_outer_limit = max_val_for_scaling + 5.0
    plot_max_r = 5.0 * math.floor(target_outer_limit / 5.0)

    if plot_max_r <= plot_min_r: plot_max_r = plot_min_r + 10.0
    ax.set_ylim(plot_min_r, plot_max_r)

    current_plot_radial_span = plot_max_r - plot_min_r
    if current_plot_radial_span <= 0: current_plot_radial_span = 10.0
    text_offset_magnitude_for_labels = current_plot_radial_span * 0.02

    num_dranges = len(drange_order)
    total_sector_width = (2 * np.pi / num_directions)
    slot_width_for_each_drange = total_sector_width / (num_dranges + 1.5)
    std_dev_range_bar_visual_width = slot_width_for_each_drange * 0.95
    mean_bar_visual_width = std_dev_range_bar_visual_width * 0.6

    for i, drange_cat in enumerate(drange_order):
        mean_snr_series = raw_mean_values_df.loc[:, drange_cat].reindex(direction_order)
        plot_snr_means = mean_snr_series.fillna(bar_start_baseline_snr).values
        
        std_dev_series = raw_std_values_df.loc[:, drange_cat].reindex(direction_order)
        plot_snr_stds = std_dev_series.fillna(0).values
            
        angle_offset = (i - (num_dranges - 1) / 2.0) * slot_width_for_each_drange
        current_bar_angles = angles_radians + angle_offset

        km_range_text = drange_definitions.get(drange_cat, "")
        full_legend_label = f"{drange_cat} {km_range_text}".strip()

        std_range_bar_bottom_values = plot_snr_means - plot_snr_stds
        std_range_bar_height_values = 2 * plot_snr_stds
        std_range_color = 'lightgray'
        ax.bar(current_bar_angles, std_range_bar_height_values,
               width=std_dev_range_bar_visual_width, alpha=0.5,     
               bottom=std_range_bar_bottom_values, color=std_range_color,
               zorder=2, edgecolor='darkgray', linewidth=0.5)

        mean_bar_heights = plot_snr_means - bar_start_baseline_snr
        mean_bars_patches = ax.bar(current_bar_angles, mean_bar_heights,
                                   width=mean_bar_visual_width, label=full_legend_label,
                                   alpha=0.85, bottom=bar_start_baseline_snr, zorder=3)

        for idx in range(len(plot_snr_means)):
            mean_val = plot_snr_means[idx]
            std_val = plot_snr_stds[idx]
            angle_val = current_bar_angles[idx]

            mean_bar_length = mean_val - bar_start_baseline_snr
            if abs(mean_bar_length) > 0.01:
                mean_label_offset_sign = np.sign(mean_bar_length)
                if mean_label_offset_sign == 0: mean_label_offset_sign = 1
                text_pos_mean = mean_val + (mean_label_offset_sign * text_offset_magnitude_for_labels)
                ax.text(angle_val, text_pos_mean, f'{mean_val:.1f}',
                        ha='center', va='bottom' if mean_label_offset_sign >= 0 else 'top',
                        fontsize=7, color='black', zorder=4)

            if std_val > 0.01:
                lower_bound = mean_val - std_val
                upper_bound = mean_val + std_val
                std_label_font_size = 6
                std_label_color = 'darkslateblue'

                text_pos_upper = upper_bound + text_offset_magnitude_for_labels
                ax.text(angle_val, text_pos_upper, f'{upper_bound:.1f}',
                        ha='center', va='bottom', fontsize=std_label_font_size,
                        color=std_label_color, zorder=4)

                text_pos_lower = lower_bound - text_offset_magnitude_for_labels
                ax.text(angle_val, text_pos_lower, f'{lower_bound:.1f}',
                        ha='center', va='top', fontsize=std_label_font_size,
                        color=std_label_color, zorder=4)

    ax.set_xticks(angles_radians)
    ax.set_xticklabels(direction_order, fontsize=10)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    shifted_grid_degrees = np.array([337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5])
    shifted_grid_radians = np.deg2rad(shifted_grid_degrees)
    for angle_rad in shifted_grid_radians:
        ax.plot([angle_rad, angle_rad], [ax.get_ylim()[0], ax.get_ylim()[1]], color='gray', linestyle='--', linewidth=0.7)
    if plot_min_r <= 0 <= plot_max_r:
        ax.plot(np.append(angles_radians, angles_radians[0]), [0]*(num_directions+1), color='grey', linestyle='-', linewidth=0.8)
    if plot_min_r <= bar_start_baseline_snr <= plot_max_r:
        ax.plot(np.append(angles_radians, angles_radians[0]), [bar_start_baseline_snr]*(num_directions+1), color='blue', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_title("Average WSPR SNR by Direction and Distance (±Std Dev of SNR)", va='bottom', fontsize=16, pad=25)
    ax.legend(title="Distance Range (km)", loc="upper right", bbox_to_anchor=(1.2, 1.05), fontsize=9, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"\nAverage SNR polar plot saved as {output_filename}")

## -----------------------------------------------------------------------------
## Plot 2: SNR Slope Analysis Functions
## -----------------------------------------------------------------------------

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


def create_snr_slope_polar_plot(df_processed_slopes, output_filename="wspr_slope_polar_plot.png"):
    if df_processed_slopes is None or df_processed_slopes.empty:
        print("No slope data to create slope plot.")
        return

    df_plot = df_processed_slopes.copy()
    required_cols = ['slope', 'map', 'km', 'Reporter']
    if not all(col in df_plot.columns for col in required_cols):
        print(f"Missing required columns for slope plot: {required_cols}")
        return

    bins_km = [0, 800, 4000, 8000, 13000, np.inf]
    labels_drange = ['NEAR', 'MID', 'LONG', 'VLONG', 'VVLONG']
    direction_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    df_plot['drange'] = pd.cut(df_plot['km'], bins=bins_km, labels=labels_drange, right=True, include_lowest=True)
    df_plot.dropna(subset=['drange'], inplace=True)
    df_plot['drange'] = pd.Categorical(df_plot['drange'], categories=labels_drange, ordered=True)
    df_plot['map'] = pd.Categorical(df_plot['map'], categories=direction_order, ordered=True)

    slope_aggregated_data = df_plot.groupby(['map', 'drange'], observed=False).agg(
        avg_slope=('slope', 'mean'),
        std_of_slopes=('slope', 'std'),
        reporter_count=('Reporter', 'nunique')
    ).reset_index()

    avg_slope_pivot = slope_aggregated_data.pivot_table(index='map', columns='drange', values='avg_slope')
    avg_slope_pivot = avg_slope_pivot.reindex(index=direction_order).reindex(columns=labels_drange)

    std_of_slopes_pivot = slope_aggregated_data.pivot_table(index='map', columns='drange', values='std_of_slopes')
    std_of_slopes_pivot = std_of_slopes_pivot.reindex(index=direction_order).reindex(columns=labels_drange).fillna(0)

    if avg_slope_pivot.isnull().all().all():
        print("Not enough aggregated slope data for plot after pivoting.")
        return

    slope_plot_baseline = 0.0
    radial_axis_padding_slope = 0.5

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))
    num_directions = len(direction_order)
    angles_radians = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

    effective_std_for_ylim_slopes = std_of_slopes_pivot.fillna(0)
    min_data_points_slopes = avg_slope_pivot - effective_std_for_ylim_slopes
    max_data_points_slopes = avg_slope_pivot + effective_std_for_ylim_slopes
    true_min_slope_extent = min_data_points_slopes.min().min(skipna=True)
    true_max_slope_extent = max_data_points_slopes.max().max(skipna=True)

    min_val_for_scaling_slopes = slope_plot_baseline
    if pd.notna(true_min_slope_extent): min_val_for_scaling_slopes = true_min_slope_extent
    max_val_for_scaling_slopes = slope_plot_baseline
    if pd.notna(true_max_slope_extent): max_val_for_scaling_slopes = true_max_slope_extent
        
    plot_min_r_slopes = min(min_val_for_scaling_slopes, slope_plot_baseline) - radial_axis_padding_slope
    plot_max_r_slopes = max(max_val_for_scaling_slopes, slope_plot_baseline) + radial_axis_padding_slope

    if abs(plot_max_r_slopes - plot_min_r_slopes) < 1.0:
        plot_min_r_slopes -= 0.5
        plot_max_r_slopes += 0.5
    if plot_max_r_slopes <= plot_min_r_slopes:
        plot_max_r_slopes = plot_min_r_slopes + 1.0
    
    ax.set_ylim(plot_min_r_slopes, plot_max_r_slopes)

    current_plot_radial_span_slopes = plot_max_r_slopes - plot_min_r_slopes
    if current_plot_radial_span_slopes <= 0: current_plot_radial_span_slopes = 1.0 
    text_offset_magnitude_slopes = current_plot_radial_span_slopes * 0.03

    num_dranges = len(labels_drange)
    total_sector_width = (2 * np.pi / num_directions)
    slot_width_for_each_drange = total_sector_width / (num_dranges + 1.5)
    slope_std_range_bar_visual_width = slot_width_for_each_drange * 0.95
    avg_slope_bar_visual_width = slope_std_range_bar_visual_width * 0.6

    drange_legend_definitions = {
        'NEAR': '(0-800 km)', 'MID': '(800-4000 km)', 
        'LONG': '(4000-8000 km)', 'VLONG': '(8000-13000 km)', 
        'VVLONG': '(13000+ km)'
    }

    for i, drange_cat in enumerate(labels_drange):
        plot_avg_slopes = avg_slope_pivot.loc[:, drange_cat].reindex(direction_order).fillna(slope_plot_baseline).values
        plot_std_of_slopes = std_of_slopes_pivot.loc[:, drange_cat].reindex(direction_order).fillna(0).values
            
        angle_offset = (i - (num_dranges - 1) / 2.0) * slot_width_for_each_drange
        current_bar_angles = angles_radians + angle_offset

        km_range_text = drange_legend_definitions.get(drange_cat, "")
        full_legend_label = f"{drange_cat} {km_range_text}".strip()

        slope_std_range_bar_bottom = plot_avg_slopes - plot_std_of_slopes
        slope_std_range_bar_height = 2 * plot_std_of_slopes
        std_range_color = 'lightsteelblue'
        ax.bar(current_bar_angles, slope_std_range_bar_height,
               width=slope_std_range_bar_visual_width, alpha=0.5,      
               bottom=slope_std_range_bar_bottom, color=std_range_color,
               zorder=2, edgecolor='slategray', linewidth=0.5)

        avg_slope_bars = ax.bar(current_bar_angles, plot_avg_slopes, 
                                width=avg_slope_bar_visual_width, label=full_legend_label,
                                alpha=0.85, bottom=slope_plot_baseline, zorder=3)

        for idx in range(len(plot_avg_slopes)):
            avg_slope_val = plot_avg_slopes[idx]
            angle = current_bar_angles[idx]
            
            # Check if the original value before fillna was NaN for avg_slope_pivot
            original_avg_slope_val = avg_slope_pivot.loc[direction_order[idx], drange_cat]

            if pd.notna(original_avg_slope_val): # Only label if there was an actual avg slope calculated
                offset_sign = np.sign(avg_slope_val)
                if offset_sign == 0 and avg_slope_val == 0: # True zero slope
                     offset_sign = 1 # Default offset direction if slope is exactly 0
                elif offset_sign == 0 and avg_slope_val !=0: # Should not happen with np.sign
                     offset_sign = np.sign(avg_slope_val + 1e-9) # Handle potential true zero causing issues

                text_radial_pos = avg_slope_val + (offset_sign * text_offset_magnitude_slopes)
                ax.text(angle, text_radial_pos, f'{avg_slope_val:.2f}', 
                        ha='center', va='bottom' if offset_sign >=0 else 'top',
                        fontsize=6, color='black', zorder=4)

    ax.set_xticks(angles_radians)
    ax.set_xticklabels(direction_order, fontsize=10)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    shifted_grid_degrees = np.array([337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5])
    shifted_grid_radians = np.deg2rad(shifted_grid_degrees)
    for angle_rad in shifted_grid_radians:
        ax.plot([angle_rad, angle_rad], [ax.get_ylim()[0], ax.get_ylim()[1]], color='gray', linestyle='--', linewidth=0.7)
    if plot_min_r_slopes <= 0 <= plot_max_r_slopes: # Use slope plot limits
        ax.plot(np.append(angles_radians, angles_radians[0]), [0]*(num_directions+1), color='black', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.set_title("Average SNR Slope by Direction and Distance (±StdDev of Slopes)", va='bottom', fontsize=14, pad=25)
    ax.legend(title="Distance Range (km)", loc="upper right", bbox_to_anchor=(1.22, 1.05), fontsize=9, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"\nSNR Slope polar plot saved as {output_filename}")


## -----------------------------------------------------------------------------
## Main Orchestration
## -----------------------------------------------------------------------------

def main():
    try:
        # 1. Load and Prepare Initial Data
        df_wspr = open_wspr_file()
        if df_wspr is None or df_wspr.empty: return
        
        df_wspr_processed = add_wspr_dimensions(df_wspr)
        if df_wspr_processed is None or df_wspr_processed.empty: return

        # For now, we'll proceed without GOES data join for simplicity of focusing on WSPR plots
        # If GOES data is needed, uncomment and ensure df_merged is used below
        # df_goes = open_goes_xray_file()
        # if df_goes is None or df_goes.empty: 
        #     print("Proceeding without GOES X-ray data.")
        #     df_merged = df_wspr_processed.copy() # Use processed WSPR data directly
        # else:
        #     df_merged = join_wspr_with_goes(df_wspr_processed, df_goes)
        #     if df_merged is None or df_merged.empty: 
        #         print("Failed to join or merged data is empty. Using processed WSPR data for plots.")
        #         df_merged = df_wspr_processed.copy()
        
        df_to_plot = df_wspr_processed.copy() # Using processed WSPR data

        # 2. Create Plot 1: Average SNR Polar Plot
        print("\n--- Generating Average SNR Polar Plot ---")
        create_polar_plot(df_to_plot.copy(), "wspr_avg_snr_polar_plot.png")

        # 3. Calculate Reporter-Specific SNR Trends for the Second Plot
        print("\n--- Calculating Reporter SNR Trends for Slope Plot ---")
        df_reporter_trends = calculate_reporter_snr_trends(df_to_plot.copy()) 

        # 4. Create Plot 2: SNR Slope Polar Plot
        if df_reporter_trends is not None and not df_reporter_trends.empty:
            print("\n--- Generating SNR Slope Polar Plot ---")
            create_snr_slope_polar_plot(df_reporter_trends, "wspr_slope_polar_plot.png")
        else:
            print("No reporter trend data available to generate SNR slope polar plot.")

    except FileNotFoundError as fnf_error:
        print(f"Error: File not found. Please ensure '{fnf_error.filename}' is in the correct location.")
    except pd.errors.EmptyDataError as ede_error:
        print(f"Error: One of the data files is empty. ({ede_error})")
    except KeyError as ke_error:
        print(f"Error: A required column is missing from the data: {ke_error}. Check file headers and content.")
    except Exception as e:
        print(f"An unexpected error occurred in main: {repr(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()