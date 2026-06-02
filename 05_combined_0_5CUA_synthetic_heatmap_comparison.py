#!/usr/bin/env python3
"""
Combined CUA + Synthetic Heatmap Comparison
============================================

Runs both the CUA envelope comparison and the standard synthetic envelope
comparison against the SAME observed data, using the SAME wrong-source
parameters (magnitude errors, distance offsets, random epicentre shifts).

This produces side-by-side heatmaps so both methods can be compared on equal terms.

Key differences between the two envelope types:
  CUA:       CUA_BASE / {mag} / {dist} / CUA_H.npy, CUA_Z.npy
  Synthetic: SYNTHETIC_BASE / R|S / {mag} / {dist} / ML_H.npy, ML_Z.npy

Station counter: A station contributes to the count for a given cell only when
AT LEAST ONE of the observed or synthetic envelope has signal (noise-vs-noise
pairs are excluded entirely).

Output: High-resolution (300 dpi) heatmap images suitable for posters.

Usage:
    python 03_combined_0_5CUA_synthetic_heatmap_comparison.py
    
Configuration:
    Edit event_ids list and config.py paths before running.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import glob
from obspy import read
from obspy.core import UTCDateTime
import rasterio
from rasterio.transform import rowcol
import requests
import json
import re
import sys

# Import configuration
from config import (
    DATA_BASE,
    DATA_DIR,
    ENVELOPES_DIR,
    CUA_BASE,
    SYNTHETIC_BASE,
    RESULTS_BASE,
    RESULTS_DIR,
    SYN_CUA_ENV,
    VS30_CSV,
    VS30_TIFF,
    MAGNITUDE_ERRORS,
    DISTANCE_OFFSETS_KM,
    TIME_WINDOWS_S,
    N_RANDOM_TRIES,
    P_WAVE_VELOCITY_KM_S,
    VS30_THRESHOLD,
    get_event_paths,
)

# ============================================================================
# CONFIGURATION  — edit these before running
# ============================================================================
event_ids = ["20190705_M7.1_Ridgecrest", "20140824_M6.0_SouthNapa", 
             "20240403_M7.4_Hualien", "20250120_M6.0_Chiayii", "20230220_M6.4_Yayladagi"]

# ============================================================================
# HELPER FUNCTIONS (shared between CUA and Synthetic processing)
# ============================================================================

def load_station_data(csv_path):
    """Load station distance information from CSV file."""
    return pd.read_csv(csv_path)


def load_vs30_data(vs30_csv_path=None):
    """Load VS30 data from CSV file."""
    if vs30_csv_path is None:
        vs30_csv_path = VS30_CSV
    return pd.read_csv(vs30_csv_path)


def get_vs30_from_tiff(tiff_path, lat, lon):
    """Get VS30 value from TIFF file at given coordinates."""
    if tiff_path is None:
        tiff_path = VS30_TIFF
    try:
        with rasterio.open(tiff_path) as src:
            row, col = rowcol(src.transform, lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                vs30_value = src.read(1)[row, col]
                if src.nodata is not None and vs30_value == src.nodata:
                    return None
                return float(vs30_value)
            else:
                return None
    except Exception as e:
        print(f"Error reading TIFF at ({lat}, {lon}): {str(e)}")
        return None


def get_vs30_from_esm(lat, lon):
    try:
        url = f"https://esm-db.eu/esmws/topography/1/query?latitude={lat}&longitude={lon}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    return None
            if isinstance(data, dict):
                vs30 = data.get('vs30', None)
                if vs30 is not None:
                    return float(vs30)
        return None
    except requests.exceptions.Timeout:
        print(f"  -> ESM database timeout at ({lat:.4f}, {lon:.4f})")
        return None
    except Exception as e:
        print(f"  -> ESM database error at ({lat:.4f}, {lon:.4f}): {str(e)}")
        return None


def get_station_site_type(station_code, vs30_df, station_info=None, tiff_path=None, allow_default=True, skip_lookup=False):
    """
    Determine station site type (R=rock, S=soil) based on VS30 value.

    Uses threshold of 450 m/s (VS30 < 450 = Soil, >= 450 = Rock).

    Priority order:
    1. VS30 from CSV
    2. VS30 from TIFF file (if tiff_path provided and skip_lookup=False)
    3. VS30 from ESM database (if station_info provided and skip_lookup=False)
    4. Default to Rock (R) with VS30=500 m/s if allow_default=True
    """
    if skip_lookup:
        parts = station_code.split('.')
        if len(parts) == 2:
            network, station = parts
            station_match = vs30_df[vs30_df['Network/Station Code'].str.contains(
                f'{network}.{station}', na=False, case=False)]
            if not station_match.empty:
                vs30_value = station_match.iloc[0]['Vs30 (m/s)']
                if pd.notna(vs30_value):
                    return 'S' if vs30_value < VS30_THRESHOLD else 'R', vs30_value
        if allow_default:
            return 'R', 500.0
        return None, None

    parts = station_code.split('.')
    if len(parts) == 2:
        network, station = parts
        station_match = vs30_df[vs30_df['Network/Station Code'].str.contains(
            f'{network}.{station}', na=False, case=False)]
        if not station_match.empty:
            vs30_value = station_match.iloc[0]['Vs30 (m/s)']
            if pd.notna(vs30_value):
                return 'S' if vs30_value < 450 else 'R', vs30_value

    if tiff_path is not None and station_info is not None:
        if 'latitude' in station_info and 'longitude' in station_info:
            vs30_value = get_vs30_from_tiff(tiff_path, station_info['latitude'], station_info['longitude'])
            if vs30_value is not None:
                return 'S' if vs30_value < 450 else 'R', vs30_value

    if station_info is not None:
        if 'latitude' in station_info and 'longitude' in station_info:
            vs30_value = get_vs30_from_esm(station_info['latitude'], station_info['longitude'])
            if vs30_value is not None:
                return 'S' if vs30_value < 450 else 'R', vs30_value

    if allow_default:
        return 'R', 500.0
    return None, None


def load_envelope(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def estimate_noise_level(envelope, p_arrival_idx):
    if p_arrival_idx is None or p_arrival_idx <= 0:
        noise_window = envelope[:min(10, len(envelope) // 10)]
    else:
        noise_window = envelope[:p_arrival_idx]
    if len(noise_window) == 0:
        return 0.0
    return np.median(noise_window)


def pad_envelope_from_origin(envelope, distance_km, p_arrival_idx=None,
                              velocity_km_s=6.0, sampling_rate=1.0, noise_level=1e-5):
    p_arrival_time = distance_km / velocity_km_s
    n_pad_samples = int(np.round(p_arrival_time * sampling_rate))
    if p_arrival_idx is not None and p_arrival_idx > 0:
        envelope_from_p = envelope[p_arrival_idx:]
    else:
        envelope_from_p = envelope
    noise_level_rounded = np.round(noise_level, decimals=0) if noise_level >= 1 else noise_level
    padding = np.full(n_pad_samples, noise_level_rounded)
    return np.concatenate([padding, envelope_from_p])


def has_signal(envelope, p_arrival_idx=None, noise_threshold_factor=2.0):
    if envelope is None or len(envelope) == 0:
        return False
    if p_arrival_idx is not None and p_arrival_idx > 0:
        return True
    noise_window_size = min(10, len(envelope) // 10)
    if noise_window_size > 0:
        noise_level = np.median(envelope[:noise_window_size])
        max_amplitude = np.max(envelope)
        return max_amplitude > noise_threshold_factor * noise_level
    return False


def calculate_metrics(obs, syn, time_window_seconds=8, sampling_rate=1.0):
    if obs is None or syn is None:
        return None, None, None, None
    max_samples = int(time_window_seconds * sampling_rate)
    min_len = min(len(obs), len(syn), max_samples)
    obs = obs[:min_len]
    syn = syn[:min_len]
    obs_std = np.std(obs)
    syn_std = np.std(syn)
    if obs_std == 0 and syn_std == 0:
        return None, None, None, None
    elif obs_std == 0 or syn_std == 0:
        correlation = 0.0
    else:
        correlation = stats.pearsonr(obs, syn)[0]
        if correlation < 0:
            correlation = 0
    rmse = np.sqrt(mean_squared_error(obs, syn))
    obs_max = np.max(obs)
    syn_max = np.max(syn)
    peak_ratio = obs_max / syn_max if syn_max > 0 else 0.0
    if (obs_max + syn_max) > 0:
        # Clamp to [0, 1] to guard against floating-point underflow when
        # obs_max ≈ syn_max, which can make the expression slightly negative
        # and cause np.sqrt to return NaN.
        amplitude_fit = max(0.0, 1 - np.sqrt(((obs_max - syn_max) ** 2) / ((obs_max + syn_max) ** 2)))
    else:
        amplitude_fit = 0.0
    gof = np.sqrt(amplitude_fit * correlation)
    return correlation, rmse, peak_ratio, gof


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


# ============================================================================
# CORE HEATMAP COMPUTATION
# ============================================================================

def compute_heatmap_for_envelopes(
        stations_df, event_lat, event_lon,
        synthetic_base, observed_base,
        observed_magnitude, vs30_df,
        magnitude_errors, distance_offsets,
        component, time_window_seconds,
        tiff_path, skip_vs30_lookup,
        n_random_tries,
        envelope_filename_h, envelope_filename_z,
        p_wave_velocity=6.0,
        random_angles=None):
    """
    Generic heatmap computation that works for both CUA and Synthetic envelopes.

    For each time window `time_window_seconds` seconds after origin, only stations
    whose P-wave has already arrived are included.  The comparison uses only the
    data available up to that moment: data_duration = time_window_seconds - p_travel_time.
    Stations where data_duration < 2 s are skipped.

    Parameters
    ----------
    envelope_filename_h : str
        Filename of the horizontal synthetic envelope inside the distance folder.
    envelope_filename_z : str
        Filename of the vertical synthetic envelope inside the distance folder.
    p_wave_velocity : float
        P-wave velocity in km/s used to compute travel times (default 6.0).

    Returns
    -------
    heatmap_gof, heatmap_correlation, heatmap_peak_ratio : 2D arrays
    magnitude_errors, distance_offsets : 1D arrays
    min_stations, max_stations : 2D arrays
        Station counts per cell (counting only pairs where at least one has signal).
    """
    gof_data = [[[] for _ in range(len(distance_offsets))] for _ in range(len(magnitude_errors))]
    correlation_data = [[[] for _ in range(len(distance_offsets))] for _ in range(len(magnitude_errors))]
    peak_ratio_data = [[[] for _ in range(len(distance_offsets))] for _ in range(len(magnitude_errors))]
    count_data = np.zeros((len(magnitude_errors), len(distance_offsets)))
    min_stations = np.full((len(magnitude_errors), len(distance_offsets)), np.inf)
    max_stations = np.zeros((len(magnitude_errors), len(distance_offsets)))

    # Get available synthetic distances for the observed magnitude
    mag_str_ref = f"{observed_magnitude:.1f}"
    distance_path = os.path.join(synthetic_base, mag_str_ref)
    if not os.path.exists(distance_path):
        print(f"  Warning: Synthetic directory not found: {distance_path}")
        empty = np.full((len(magnitude_errors), len(distance_offsets)), np.nan)
        return empty, empty, empty, magnitude_errors, distance_offsets, None, None

    distances = [d for d in os.listdir(distance_path) if os.path.isdir(os.path.join(distance_path, d))]
    synthetic_distances = sorted([float(d) for d in distances if d.replace('.', '', 1).isdigit()])
    print(f"  Available synthetic distances: {synthetic_distances}")

    # Component file/folder setup
    if component == 'H':
        comp_folder = 'Horizontal_combined'
        obs_filenames = ['envelope_HLE_HLN.npy', 'envelope_HNE_HNN.npy']
        meta_filenames = ['envelope_HLE_HLN_meta.json', 'envelope_HNE_HNN_meta.json']
        syn_filename = envelope_filename_h
    else:
        comp_folder = 'Vertical'
        obs_filenames = ['envelope_HLZ.npy', 'envelope_HNZ.npy']
        meta_filenames = ['envelope_HLZ_meta.json', 'envelope_HNZ_meta.json']
        syn_filename = envelope_filename_z

    # Pre-load observed envelopes (same across all distance offsets / magnitudes)
    print(f"  Pre-loading observed envelopes for component {component}...")
    station_obs_cache = {}  # station_code -> (observed, observed_padded, obs_has_signal, p_arrival_idx, true_distance)
    for _, station in stations_df.iterrows():
        station_code = station['station_code']
        station_lat = station.get('latitude')
        station_lon = station.get('longitude')
        if station_lat is None or station_lon is None:
            continue
        true_distance = haversine_distance(event_lat, event_lon, station_lat, station_lon)

        obs_path = os.path.join(observed_base, station_code, comp_folder)
        if not os.path.exists(obs_path):
            continue

        obs_file = None
        obs_filename_found = None
        for filename in obs_filenames:
            test_file = os.path.join(obs_path, filename)
            if os.path.exists(test_file):
                obs_file = test_file
                obs_filename_found = filename
                break
        if obs_file is None:
            continue

        observed = load_envelope(obs_file)
        if observed is None:
            continue

        # Load metadata
        metadata = None
        meta_idx = obs_filenames.index(obs_filename_found)
        meta_file = os.path.join(obs_path, meta_filenames[meta_idx])
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass

        p_arrival_idx = None
        if metadata and 'p_arrival_idx_in_envelope' in metadata:
            p_arrival_idx = int(metadata['p_arrival_idx_in_envelope'])

        observed_padded = pad_envelope_from_origin(
            observed,
            distance_km=true_distance,
            p_arrival_idx=p_arrival_idx,
            velocity_km_s=6.0,
            sampling_rate=1.0
        )
        obs_has_signal = has_signal(observed, p_arrival_idx)

        station_obs_cache[station_code] = (observed_padded, obs_has_signal, p_arrival_idx, true_distance,
                                           station_lat, station_lon)

    print(f"  Cached {len(station_obs_cache)} observed envelopes.")

    # Main loop
    for dist_idx, dist_offset in enumerate(distance_offsets):
        print(f"  Distance offset {dist_offset:.0f} km ({n_random_tries} random tries)...")

        for try_idx in range(n_random_tries):
            stations_this_try = {mag_idx: set() for mag_idx in range(len(magnitude_errors))}

            # Random epicentre shift – use shared angles if provided so CUA and
            # Synthetic are evaluated at the exact same shifted epicentres.
            if random_angles is not None:
                angle = random_angles[dist_idx, try_idx]
            else:
                angle = np.random.uniform(0, 2 * np.pi)
            lat_offset = dist_offset * np.sin(angle) / 111.0
            lon_offset = dist_offset * np.cos(angle) / (111.0 * np.cos(np.radians(event_lat)))
            new_event_lat = event_lat + lat_offset
            new_event_lon = event_lon + lon_offset

            for station_code, (observed_padded, obs_has_signal, p_arrival_idx,
                               true_distance, station_lat, station_lon) in station_obs_cache.items():
                # Distance from shifted epicentre (for synthetic selection and P-wave timing)
                new_distance = haversine_distance(new_event_lat, new_event_lon, station_lat, station_lon)

                # P-wave travel time from the (wrong) shifted source
                p_wave_travel_time = new_distance / p_wave_velocity

                # Skip stations whose P-wave has not yet arrived at this time window
                if p_wave_travel_time > time_window_seconds:
                    continue

                # How many seconds of signal are available at this time since origin
                data_duration = time_window_seconds - p_wave_travel_time
                if data_duration < 2.0:
                    continue

                closest_synthetic_dist = min(synthetic_distances, key=lambda x: abs(float(x) - new_distance))

                for mag_idx, mag_error in enumerate(magnitude_errors):
                    test_magnitude = observed_magnitude + mag_error
                    if test_magnitude < 4.0 - 1e-9 or test_magnitude > 8.0 + 1e-9:
                        continue

                    mag_str = f"{test_magnitude:.1f}"
                    synthetic_path = os.path.join(synthetic_base, mag_str,
                                                  str(int(closest_synthetic_dist)), syn_filename)
                    if not os.path.exists(synthetic_path):
                        continue

                    synthetic = load_envelope(synthetic_path)
                    if synthetic is None:
                        continue

                    syn_has_signal = has_signal(synthetic, p_arrival_idx=None)

                    # Only include pairs where at least one envelope has signal
                    if not (obs_has_signal or syn_has_signal):
                        continue

                    synthetic_padded = pad_envelope_from_origin(
                        synthetic,
                        distance_km=closest_synthetic_dist,
                        p_arrival_idx=None,
                        velocity_km_s=p_wave_velocity,
                        sampling_rate=1.0
                    )

                    # Use only the data available so far (time since origin minus travel time)
                    metrics = calculate_metrics(observed_padded, synthetic_padded,
                                               time_window_seconds=data_duration)

                    if metrics[0] is not None:
                        gof_data[mag_idx][dist_idx].append(metrics[3])
                        correlation_data[mag_idx][dist_idx].append(metrics[0])
                        peak_ratio_data[mag_idx][dist_idx].append(metrics[2])
                        count_data[mag_idx, dist_idx] += 1
                        stations_this_try[mag_idx].add(station_code)

            # Update min/max station counts per try
            for mag_idx, sta_set in stations_this_try.items():
                n = len(sta_set)
                if n > 0:
                    min_stations[mag_idx, dist_idx] = min(min_stations[mag_idx, dist_idx], n)
                    max_stations[mag_idx, dist_idx] = max(max_stations[mag_idx, dist_idx], n)

        print(f"  Distance offset {dist_offset:.0f} km complete: "
              f"{int(count_data[:, dist_idx].sum())} total comparisons across {n_random_tries} tries")

    # Calculate median values
    heatmap_gof = np.full((len(magnitude_errors), len(distance_offsets)), np.nan)
    heatmap_correlation = np.full((len(magnitude_errors), len(distance_offsets)), np.nan)
    heatmap_peak_ratio = np.full((len(magnitude_errors), len(distance_offsets)), np.nan)

    for mag_idx in range(len(magnitude_errors)):
        for dist_idx in range(len(distance_offsets)):
            if len(gof_data[mag_idx][dist_idx]) > 0:
                heatmap_gof[mag_idx, dist_idx] = np.nanmedian(gof_data[mag_idx][dist_idx])
                heatmap_correlation[mag_idx, dist_idx] = np.nanmedian(correlation_data[mag_idx][dist_idx])
                heatmap_peak_ratio[mag_idx, dist_idx] = np.nanmedian(peak_ratio_data[mag_idx][dist_idx])

    # Clean up station counts
    min_stations = np.where(count_data > 0, min_stations, np.nan)
    min_stations = np.where(min_stations == np.inf, np.nan, min_stations)
    max_stations = np.where(count_data > 0, max_stations, np.nan)

    print(f"  Heatmap complete. Valid cells: {np.sum(~np.isnan(heatmap_gof))} / {heatmap_gof.size}")

    return (heatmap_gof, heatmap_correlation, heatmap_peak_ratio,
            magnitude_errors, distance_offsets, min_stations, max_stations)


# ============================================================================
# SIDE-BY-SIDE HEATMAP PLOTTING
# ============================================================================

def plot_combined_heatmaps(heatmap_gof_cua, heatmap_gof_syn,
                           heatmap_corr_cua, heatmap_corr_syn,
                           heatmap_peak_cua, heatmap_peak_syn,
                           magnitude_errors, distance_offsets,
                           observed_magnitude, output_dir,
                           component='H', time_window=None,
                           min_stations_cua=None, max_stations_cua=None,
                           min_stations_syn=None, max_stations_syn=None):
    """
    Plot side-by-side heatmaps comparing CUA and Synthetic envelopes.
    One figure per metric (GoF, Correlation, Peak Ratio), each containing
    a 1×2 grid: left = CUA, right = Synthetic.

    Saves at 300 dpi for poster quality.
    """
    # Trim magnitude axis to rows with valid templates (template mag <= 8.0)
    MAX_TEMPLATE_MAG = 8.0
    valid_rows = (observed_magnitude + magnitude_errors) <= MAX_TEMPLATE_MAG + 1e-9
    magnitude_errors  = magnitude_errors[valid_rows]
    heatmap_gof_cua   = heatmap_gof_cua[valid_rows]
    heatmap_gof_syn   = heatmap_gof_syn[valid_rows]
    heatmap_corr_cua  = heatmap_corr_cua[valid_rows]
    heatmap_corr_syn  = heatmap_corr_syn[valid_rows]
    heatmap_peak_cua  = heatmap_peak_cua[valid_rows]
    heatmap_peak_syn  = heatmap_peak_syn[valid_rows]

    cmap = 'viridis'
    heatmap_dir = os.path.join(output_dir, 'combined_heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    time_suffix = f'_{time_window}s' if time_window else ''

    metrics_info = [
        (heatmap_gof_cua, heatmap_gof_syn, 'Goodness of Fit (GoF)', 'gof', (0, 100),
         min_stations_cua, max_stations_cua, min_stations_syn, max_stations_syn),
        (heatmap_corr_cua, heatmap_corr_syn, 'Correlation', 'correlation', (0, 100),
         min_stations_cua, max_stations_cua, min_stations_syn, max_stations_syn),
        (heatmap_peak_cua, heatmap_peak_syn, 'Peak Ratio', 'peak_ratio', (0, 2),
         min_stations_cua, max_stations_cua, min_stations_syn, max_stations_syn),
    ]

    for (data_cua, data_syn, metric_name, metric_short, vrange,
         min_cua, max_cua, min_syn, max_syn) in metrics_info:

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        panels = [
            (axes[0], data_cua * 100 if vrange[1] == 100 else data_cua, 'CUA Envelopes', min_cua, max_cua),
            (axes[1], data_syn * 100 if vrange[1] == 100 else data_syn, 'Synthetic Envelopes (AZ_test)', min_syn, max_syn),
        ]

        for ax, heatmap_data, title_suffix, min_sta, max_sta in panels:
            im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap,
                           vmin=vrange[0], vmax=vrange[1],
                           origin='lower', interpolation='nearest')

            ax.set_xticks(np.arange(len(distance_offsets)))
            ax.set_xticklabels([f'{d:.0f}' for d in distance_offsets], fontsize=11)

            y_tick_indices = np.arange(0, len(magnitude_errors), 5)
            ax.set_yticks(y_tick_indices)
            ax.set_yticklabels([f'{magnitude_errors[i]:+.1f}' for i in y_tick_indices], fontsize=11)

            zero_idx = np.argmin(np.abs(magnitude_errors))
            ax.axhline(y=zero_idx, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label='True Magnitude')
            ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                       label='True Location')

            ax.set_xlabel('Distance Error (km)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Magnitude Error', fontsize=13, fontweight='bold')

            time_info = f' ({time_window}s)' if time_window else ''
            ax.set_title(f'{metric_name} – {title_suffix}{time_info}\n'
                         f'Component: {component}, True M{observed_magnitude:.1f}',
                         fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, label=metric_name + (' [%]' if vrange[1] == 100 else ''))
            cbar.ax.tick_params(labelsize=11)
            ax.legend(loc='upper right', fontsize=10)

            # Add minor grid
            ax.set_xticks(np.arange(len(distance_offsets)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(magnitude_errors)) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.4, alpha=0.3)

            # Annotate cells
            for i in range(len(magnitude_errors)):
                for j in range(len(distance_offsets)):
                    if not np.isnan(heatmap_data[i, j]):
                        metric_val = heatmap_data[i, j]
                        if (min_sta is not None and max_sta is not None
                                and not np.isnan(min_sta[i, j])):
                            min_s = int(min_sta[i, j])
                            max_s = int(max_sta[i, j])
                            ax.text(j, i - 0.28, f'{metric_val:.1f}' + ('%' if vrange[1] == 100 else ''),
                                    ha='center', va='center', color='white',
                                    fontsize=7.5, fontweight='bold')
                            ax.text(j, i + 0.28, f'[{min_s}-{max_s}]',
                                    ha='center', va='center', color='lightgray', fontsize=6)
                        else:
                            ax.text(j, i, f'{metric_val:.1f}' + ('%' if vrange[1] == 100 else ''),
                                    ha='center', va='center', color='white',
                                    fontsize=8, fontweight='bold')

            # Highlight maximum cell
            if not np.all(np.isnan(heatmap_data)):
                max_val = np.nanmax(heatmap_data)
                for max_pos in np.argwhere(heatmap_data == max_val):
                    ii, jj = max_pos
                    rect = plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False,
                                         edgecolor='red', linewidth=3, zorder=10)
                    ax.add_patch(rect)

        plt.tight_layout()

        output_path = os.path.join(
            heatmap_dir,
            f'combined_heatmap_{metric_short}_{component}{time_suffix}.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

        # Save CSVs
        for label, data in [('CUA', data_cua), ('Synthetic', data_syn)]:
            df = pd.DataFrame(data,
                              index=[f'{e:+.1f}' for e in magnitude_errors],
                              columns=[f'{d:.0f}' for d in distance_offsets])
            csv_path = os.path.join(
                heatmap_dir,
                f'combined_heatmap_{metric_short}_{label}_{component}{time_suffix}.csv'
            )
            df.to_csv(csv_path)


def plot_side_by_side_single_metric(heatmap_cua, heatmap_syn,
                                    magnitude_errors, distance_offsets,
                                    observed_magnitude, metric_name, metric_short,
                                    vrange, output_dir, component, time_window,
                                    min_sta_cua, max_sta_cua,
                                    min_sta_syn, max_sta_syn):
    """Thin wrapper around the combined plot for a single metric."""
    # Build dummy dicts for the other two metrics (not used here)
    empty = np.full_like(heatmap_cua, np.nan)
    if metric_short == 'gof':
        plot_combined_heatmaps(
            heatmap_cua, heatmap_syn,
            empty, empty, empty, empty,
            magnitude_errors, distance_offsets,
            observed_magnitude, output_dir, component, time_window,
            min_sta_cua, max_sta_cua, min_sta_syn, max_sta_syn
        )


# ============================================================================
# ENVELOPE-vs-ENVELOPE COMPARISON PLOTS
# ============================================================================

def plot_envelope_comparison_per_station(
        station_code, station_lat, station_lon,
        event_lat, event_lon,
        observed_base, cua_base, syn_base,
        observed_magnitude,
        output_dir,
        time_windows=None,
        dist_error=0.0,
        mag_error=0.0,
        shifted_event_lat=None,
        shifted_event_lon=None,
        p_wave_velocity=6.0):
    """
    For a single station and a specific (dist_error, mag_error) combination,
    create one figure with all time windows as rows and 2 components (H, Z) as
    columns.  Style mirrors plot_diagnostic_noise_padding from file 03:

      - Blue  line + blue  axvspan  → Observed  (TRUE distance)
      - Green line + green axvspan  → CUA        (wrong distance from shifted epicenter,
                                                   wrong magnitude)
      - Red   line + red   axvspan  → Synthetic  (same wrong distance/magnitude)
      - Dashed axvlines for P-wave arrivals
      - Noise region shaded from t=0 to P-arrival for each envelope

    Time axis: seconds since event origin (padded envelopes).
    Saves one file per (station, dist_error, mag_error) combination.
    """
    if time_windows is None:
        time_windows = [4, 8, 15, 30, 60]

    true_distance = haversine_distance(event_lat, event_lon, station_lat, station_lon)

    # Distance used for CUA/Synthetic lookup – from shifted (wrong) epicenter
    if shifted_event_lat is not None and shifted_event_lon is not None:
        wrong_distance = haversine_distance(shifted_event_lat, shifted_event_lon,
                                            station_lat, station_lon)
    else:
        wrong_distance = true_distance

    test_magnitude = observed_magnitude + mag_error
    if test_magnitude < 4.0 - 1e-9 or test_magnitude > 8.0 + 1e-9:
        return
    mag_str_wrong = f"{test_magnitude:.1f}"

    # ---- Resolve CUA distances ----
    cua_distance_path = os.path.join(cua_base, mag_str_wrong)
    if not os.path.exists(cua_distance_path):
        return
    cua_dist_entries = [d for d in os.listdir(cua_distance_path)
                        if os.path.isdir(os.path.join(cua_distance_path, d))]
    cua_distances = sorted([float(d) for d in cua_dist_entries if d.replace('.', '', 1).isdigit()])
    if not cua_distances:
        return
    closest_cua_dist = min(cua_distances, key=lambda x: abs(x - wrong_distance))

    # ---- Resolve Synthetic distances ----
    syn_distance_path = os.path.join(syn_base, mag_str_wrong)
    closest_syn_dist = closest_cua_dist
    if os.path.exists(syn_distance_path):
        syn_dist_entries = [d for d in os.listdir(syn_distance_path)
                            if os.path.isdir(os.path.join(syn_distance_path, d))]
        syn_distances_list = sorted([float(d) for d in syn_dist_entries
                                     if d.replace('.', '', 1).isdigit()])
        if syn_distances_list:
            closest_syn_dist = min(syn_distances_list, key=lambda x: abs(x - wrong_distance))

    components = [
        ('H', 'Horizontal_combined',
         ['envelope_HLE_HLN.npy', 'envelope_HNE_HNN.npy'],
         ['envelope_HLE_HLN_meta.json', 'envelope_HNE_HNN_meta.json'],
         'CUA_H.npy', 'ML_H.npy'),
        ('Z', 'Vertical',
         ['envelope_HLZ.npy', 'envelope_HNZ.npy'],
         ['envelope_HLZ_meta.json', 'envelope_HNZ_meta.json'],
         'CUA_Z.npy', 'ML_Z.npy'),
    ]

    # ---- Pre-load envelopes for both components ----
    comp_data = {}  # comp -> (obs_padded, cua_padded, syn_padded, p_arr_obs, p_arr_cua, p_arr_syn)
    for comp, comp_folder, obs_files, meta_files, cua_fname, syn_fname in components:
        # Observed (always TRUE distance)
        obs_path = os.path.join(observed_base, station_code, comp_folder)
        observed = None
        p_arrival_idx = None
        for fname, mfname in zip(obs_files, meta_files):
            fpath = os.path.join(obs_path, fname)
            if os.path.exists(fpath):
                observed = load_envelope(fpath)
                meta_path = os.path.join(obs_path, mfname)
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        raw_idx = meta.get('p_arrival_idx_in_envelope')
                        if raw_idx is not None:
                            p_arrival_idx = int(raw_idx)
                    except Exception:
                        pass
                break

        # CUA (wrong magnitude, wrong distance)
        cua_path = os.path.join(cua_base, mag_str_wrong, str(int(closest_cua_dist)), cua_fname)
        cua_raw = load_envelope(cua_path) if os.path.exists(cua_path) else None

        # Synthetic (wrong magnitude, wrong distance)
        syn_path = os.path.join(syn_base, mag_str_wrong, str(int(closest_syn_dist)), syn_fname)
        syn_raw = load_envelope(syn_path) if os.path.exists(syn_path) else None

        # Pad all envelopes from origin
        obs_padded = (pad_envelope_from_origin(observed, true_distance, p_arrival_idx,
                                               velocity_km_s=p_wave_velocity)
                      if observed is not None else None)
        cua_padded = (pad_envelope_from_origin(cua_raw, closest_cua_dist, None,
                                               velocity_km_s=p_wave_velocity)
                      if cua_raw is not None else None)
        syn_padded = (pad_envelope_from_origin(syn_raw, closest_syn_dist, None,
                                               velocity_km_s=p_wave_velocity)
                      if syn_raw is not None else None)

        p_arr_obs = int(round(true_distance / p_wave_velocity))
        p_arr_cua = int(round(closest_cua_dist / p_wave_velocity))
        p_arr_syn = int(round(closest_syn_dist / p_wave_velocity))

        comp_data[comp] = (obs_padded, cua_padded, syn_padded,
                           p_arr_obs, p_arr_cua, p_arr_syn)

    if not comp_data:
        return
    if all(v[0] is None for v in comp_data.values()):
        return  # no observed data for this station

    env_dir = os.path.join(output_dir, 'envelope_comparisons')
    os.makedirs(env_dir, exist_ok=True)

    # ---- One figure: rows = time windows, cols = components ----
    n_tw = len(time_windows)
    fig, axes = plt.subplots(n_tw, 2, figsize=(14, 4 * n_tw), squeeze=False)

    for tw_idx, tw in enumerate(time_windows):
        display_tw = 35 if tw == 60 else tw
        max_samples = display_tw + 1

        for col_idx, (comp, comp_folder, *_) in enumerate(components):
            ax = axes[tw_idx, col_idx]
            (obs_padded, cua_padded, syn_padded,
             p_arr_obs, p_arr_cua, p_arr_syn) = comp_data[comp]

            for env, color, label, p_arr in [
                (obs_padded, 'blue',  f'Observed (TRUE {true_distance:.0f} km)',           p_arr_obs),
                (cua_padded, 'green', f'CUA (M{test_magnitude:.1f}, {closest_cua_dist:.0f} km)', p_arr_cua),
                (syn_padded, 'red',   f'Syn (M{test_magnitude:.1f}, {closest_syn_dist:.0f} km)', p_arr_syn),
            ]:
                if env is None:
                    continue
                min_len = min(len(env), max_samples)
                time_axis = np.arange(min_len)
                ax.plot(time_axis, env[:min_len], color=color, linewidth=2,
                        label=label, alpha=0.8)
                if p_arr < min_len:
                    ax.axvspan(0, p_arr, alpha=0.15, color=color)
                    ax.axvline(p_arr, color=color, linestyle='--', linewidth=2, alpha=0.7)

            ax.set_xlabel('Time since origin (s)', fontsize=10)
            ax.set_ylabel('Amplitude (m/s²)', fontsize=10)
            ax.set_title(f'{comp}  |  {tw}s', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, display_tw)

    fig.suptitle(
        f'Station {station_code}  |  True dist {true_distance:.0f} km  |  M{observed_magnitude:.1f}\n'
        f'Dist error: {dist_error:.0f} km  |  Mag error: {mag_error:+.1f}  '
        f'(Template: dist {wrong_distance:.0f} km, M{test_magnitude:.1f})',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(
        env_dir,
        f'diagnostic_3way_{station_code}_dist{dist_error:.0f}km_mag{mag_error:+.1f}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_path}")


def plot_all_station_envelope_comparisons(
        stations_df, event_lat, event_lon,
        observed_base, cua_base, syn_base,
        observed_magnitude, output_dir,
        time_windows=None,
        n_stations=20,
        magnitude_errors=None,
        distance_errors=None,
        p_wave_velocity=6.0):
    """
    Select n_stations stations stratified across distance ranges, then for each
    produce envelope comparison plots for all combinations of magnitude_errors ×
    distance_errors (matching the diagnostic style of file 03).

    For each distance_error a single random epicenter shift is drawn (same angle
    applied to all magnitude errors for that offset), consistent with the heatmap
    computation logic.
    """
    if time_windows is None:
        time_windows = [4, 8, 15, 30, 60]
    if magnitude_errors is None:
        magnitude_errors = [-1.0, -0.5, 0.0, 0.5, 1.0]
    if distance_errors is None:
        distance_errors = [0, 20, 50, 100]

    print(f"\n{'=' * 70}")
    print("ENVELOPE-vs-ENVELOPE COMPARISON PLOTS (stratified stations)")
    print(f"  n_stations={n_stations}  |  dist errors: {distance_errors} km  "
          f"|  mag errors: {magnitude_errors}")
    print(f"{'=' * 70}")

    # ---- Compute true distances; keep only stations with lat/lon ----
    valid_rows = []
    for _, station in stations_df.iterrows():
        lat = station.get('latitude')
        lon = station.get('longitude')
        if lat is None or lon is None:
            continue
        dist = haversine_distance(event_lat, event_lon, lat, lon)
        valid_rows.append((station, dist))

    if not valid_rows:
        print("  No valid stations with coordinates found.")
        return

    # ---- Stratified sampling: divide into n_stations equal-quantile bins ----
    valid_rows.sort(key=lambda x: x[1])  # sort by distance
    n_valid = len(valid_rows)
    actual_n = min(n_stations, n_valid)
    bin_size = n_valid / actual_n
    rng = np.random.default_rng(seed=42)  # reproducible selection
    selected = []
    for i in range(actual_n):
        bin_start = int(i * bin_size)
        bin_end = max(int((i + 1) * bin_size), bin_start + 1)
        bin_end = min(bin_end, n_valid)
        idx = int(rng.integers(bin_start, bin_end))
        selected.append(valid_rows[idx])

    dist_vals = [d for _, d in selected]
    print(f"  Selected {len(selected)} of {n_valid} valid stations  "
          f"(dist {min(dist_vals):.0f}–{max(dist_vals):.0f} km)")

    # ---- Main loop: station → dist_error → mag_error ----
    n_plotted = 0
    for station, true_dist in selected:
        station_code = station['station_code']
        station_lat = float(station['latitude'])
        station_lon = float(station['longitude'])
        print(f"\n  Station {station_code}  (dist {true_dist:.0f} km)")

        for dist_error in distance_errors:
            # One random epicenter shift per distance error value
            if dist_error == 0:
                shifted_lat, shifted_lon = event_lat, event_lon
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                lat_off = dist_error * np.sin(angle) / 111.0
                lon_off = (dist_error * np.cos(angle)
                           / (111.0 * np.cos(np.radians(event_lat))))
                shifted_lat = event_lat + lat_off
                shifted_lon = event_lon + lon_off

            for mag_error in magnitude_errors:
                print(f"    dist_error={dist_error:.0f} km, mag_error={mag_error:+.1f}")
                plot_envelope_comparison_per_station(
                    station_code=station_code,
                    station_lat=station_lat,
                    station_lon=station_lon,
                    event_lat=event_lat,
                    event_lon=event_lon,
                    observed_base=observed_base,
                    cua_base=cua_base,
                    syn_base=syn_base,
                    observed_magnitude=observed_magnitude,
                    output_dir=output_dir,
                    time_windows=time_windows,
                    dist_error=dist_error,
                    mag_error=mag_error,
                    shifted_event_lat=shifted_lat,
                    shifted_event_lon=shifted_lon,
                    p_wave_velocity=p_wave_velocity,
                )
        n_plotted += 1

    print(f"\n  Envelope comparison plots complete for {n_plotted} stations.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ============================================================================
    # CONFIGURATION  (shared across all events)
    # ============================================================================
    import re, json

    TIME_WINDOWS    = [4, 8, 15, 30, 60]
    N_RANDOM_TRIES  = 10
    SKIP_VS30_LOOKUP = True
    DISTANCE_OFFSETS = np.array([0, 5, 10, 20, 35, 50, 75, 100, 130])
    MAGNITUDE_ERRORS = np.arange(-1.5, 1.55, 0.1)

    # Both CUA and synthetic templates now live in the same unified tree:
    # SYN_CUA_ENV / R|S / mag / dist / {CUA_H.npy, CUA_Z.npy, ML_H.npy, ML_Z.npy}
    cua_dirs = {
        'R': os.path.join(SYN_CUA_ENV, 'R'),
        'S': os.path.join(SYN_CUA_ENV, 'S'),
    }
    syn_dirs = {
        'R': os.path.join(SYN_CUA_ENV, 'R'),
        'S': os.path.join(SYN_CUA_ENV, 'S'),
    }

    vs30_csv  = VS30_CSV
    vs30_tiff = VS30_TIFF if (VS30_TIFF and os.path.exists(VS30_TIFF)) else None

    vs30_df = load_vs30_data(vs30_csv)

    # =========================================================================
    # EVENT LOOP
    # =========================================================================
    for event_id in event_ids:
        observed_magnitude = float(event_id.split('_')[1].replace('M', ''))

        event_name = event_id.split('_')[-1]
        short_name = re.sub(r'(?<!^)(?=[A-Z])', '_', event_name).lower()

        station_csv   = os.path.join(DATA_DIR,      f'station_distance_table_{event_id}.csv')
        observed_base = os.path.join(ENVELOPES_DIR,  f'{short_name}_envelope')
        output_dir    = os.path.join(RESULTS_BASE, f'{event_id}_combined_CUA_synthetic_heatmap_multiwindow')

        # -------------------------------------------------------------------------
        # Load event info
        # -------------------------------------------------------------------------
        event_json_file = os.path.join(DATA_DIR, f'{event_id}.json')
        event_lat = event_lon = event_origin_time = None
        if os.path.exists(event_json_file):
            try:
                with open(event_json_file, 'r') as f:
                    event_data = json.load(f)
                if 'events' in event_data and len(event_data['events']) > 0:
                    ev = event_data['events'][0]
                    event_origin_time = ev.get('time')
                    event_lat = ev.get('latitude') or ev.get('lat')
                    event_lon = ev.get('longitude') or ev.get('lon')
                    print(f"Event origin: {event_origin_time}")
                    print(f"Event location: {event_lat:.4f}°N, {event_lon:.4f}°E")
            except Exception as e:
                print(f"Warning: could not load event JSON: {e}")

        # -------------------------------------------------------------------------
        # Validate paths
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("COMBINED CUA + SYNTHETIC HEATMAP COMPARISON")
        print("=" * 70)
        print(f"Event: {event_id}  (M{observed_magnitude})")
        print(f"Time windows: {TIME_WINDOWS} s  |  Distance offsets: {DISTANCE_OFFSETS} km")
        print(f"Random tries per offset: {N_RANDOM_TRIES}")

        if not os.path.exists(station_csv):
            print(f"ERROR: Station CSV not found: {station_csv} – skipping.")
            continue
        if not os.path.exists(observed_base):
            print(f"ERROR: Observed envelope directory not found: {observed_base} – skipping.")
            continue
        if event_lat is None or event_lon is None:
            print("ERROR: Event location not available – skipping.")
            continue

        os.makedirs(output_dir, exist_ok=True)

        stations_df = load_station_data(station_csv)
        print(f"\nLoaded {len(stations_df)} stations, {len(vs30_df)} VS30 records.")

        cua_base = cua_dirs['R']
        syn_base = syn_dirs['R']
        print(f"\nCUA synthetic base:       {cua_base}")
        print(f"Standard synthetic base:  {syn_base}")

        # -------------------------------------------------------------------------
        # Envelope-vs-envelope comparison plots (observed + CUA + synthetic)
        # -------------------------------------------------------------------------
        plot_all_station_envelope_comparisons(
            stations_df=stations_df,
            event_lat=event_lat,
            event_lon=event_lon,
            observed_base=observed_base,
            cua_base=cua_base,
            syn_base=syn_base,
            observed_magnitude=observed_magnitude,
            output_dir=output_dir,
            time_windows=TIME_WINDOWS,
        )

        # -------------------------------------------------------------------------
        # Pre-generate shared random epicentre angles so CUA and Synthetic are
        # evaluated at identical shifted locations for every (dist_offset, try).
        # -------------------------------------------------------------------------
        random_angles = np.random.uniform(
            0, 2 * np.pi, size=(len(DISTANCE_OFFSETS), N_RANDOM_TRIES))

        # -------------------------------------------------------------------------
        # Compute heatmaps for each component and each time window
        # -------------------------------------------------------------------------
        for component in ['H', 'Z']:
            print(f"\n{'=' * 70}")
            print(f"Processing component: {component}")
            print(f"{'=' * 70}")

            for tw in TIME_WINDOWS:
                print(f"\n{'─' * 70}")
                print(f"  Time window: {tw}s after origin  |  Component: {component}")
                print(f"{'─' * 70}")

                # --- CUA ---
                print(f"\n  [CUA] Computing heatmap ({tw}s, {component})...")
                (heatmap_gof_cua, heatmap_corr_cua, heatmap_peak_cua,
                 mag_errors, dist_offsets,
                 min_sta_cua, max_sta_cua) = compute_heatmap_for_envelopes(
                    stations_df=stations_df,
                    event_lat=event_lat,
                    event_lon=event_lon,
                    synthetic_base=cua_base,
                    observed_base=observed_base,
                    observed_magnitude=observed_magnitude,
                    vs30_df=vs30_df,
                    magnitude_errors=MAGNITUDE_ERRORS,
                    distance_offsets=DISTANCE_OFFSETS,
                    component=component,
                    time_window_seconds=tw,
                    tiff_path=vs30_tiff,
                    skip_vs30_lookup=SKIP_VS30_LOOKUP,
                    n_random_tries=N_RANDOM_TRIES,
                    envelope_filename_h='CUA_H.npy',
                    envelope_filename_z='CUA_Z.npy',
                    random_angles=random_angles,
                )

                # --- Standard (AZ_test) Synthetics ---
                print(f"\n  [Synthetic] Computing heatmap ({tw}s, {component})...")
                (heatmap_gof_syn, heatmap_corr_syn, heatmap_peak_syn,
                 _, _,
                 min_sta_syn, max_sta_syn) = compute_heatmap_for_envelopes(
                    stations_df=stations_df,
                    event_lat=event_lat,
                    event_lon=event_lon,
                    synthetic_base=syn_base,
                    observed_base=observed_base,
                    observed_magnitude=observed_magnitude,
                    vs30_df=vs30_df,
                    magnitude_errors=MAGNITUDE_ERRORS,
                    distance_offsets=DISTANCE_OFFSETS,
                    component=component,
                    time_window_seconds=tw,
                    tiff_path=vs30_tiff,
                    skip_vs30_lookup=SKIP_VS30_LOOKUP,
                    n_random_tries=N_RANDOM_TRIES,
                    envelope_filename_h='ML_H.npy',
                    envelope_filename_z='ML_Z.npy',
                    random_angles=random_angles,
                )

                # --- Plot side-by-side heatmaps ---
                print(f"\n  [Plot] Saving combined heatmaps ({tw}s, {component})...")
                plot_combined_heatmaps(
                    heatmap_gof_cua=heatmap_gof_cua,
                    heatmap_gof_syn=heatmap_gof_syn,
                    heatmap_corr_cua=heatmap_corr_cua,
                    heatmap_corr_syn=heatmap_corr_syn,
                    heatmap_peak_cua=heatmap_peak_cua,
                    heatmap_peak_syn=heatmap_peak_syn,
                    magnitude_errors=mag_errors,
                    distance_offsets=dist_offsets,
                    observed_magnitude=observed_magnitude,
                    output_dir=output_dir,
                    component=component,
                    time_window=tw,
                    min_stations_cua=min_sta_cua,
                    max_stations_cua=max_sta_cua,
                    min_stations_syn=min_sta_syn,
                    max_stations_syn=max_sta_syn,
                )

        print("\n" + "=" * 70)
        print(f"Event {event_id} complete!")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

    print("\nAll events processed.")


if __name__ == '__main__':
    main()
