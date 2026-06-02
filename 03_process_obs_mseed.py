#!/usr/bin/env python3
"""
Process Observed MSEED Data
============================

This script processes raw seismic data (miniSEED files) by:
1. Filtering out traces with zeros/NaN/Inf
2. Removing instrument response
3. Demeaning and detrending
4. Filtering by distance
5. Removing clipped signals using enhanced detection
6. Saving only HN/HL channels

Output:
- processed.ms: Processed traces in miniSEED format
- processed_stations.txt: List of successfully processed station codes
- trace_metadata.json: Metadata including absolute start times for each trace

Usage:
    python 03_process_obs_mseed.py
    
Configuration:
    Edit event_id in this script and adjust paths in config.py
"""

import obspy
from obspy import read, read_inventory
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Import configuration
from config import (
    DATA_DIR,
    PROCESSED_DIR,
    MAX_DISTANCE_KM,
    CLIP_THRESHOLD,
    ZERO_FRAC_THRESHOLD,
    get_event_paths,
)

# ============================================================================
# CONFIGURATION  — edit these before running
# ============================================================================
event_id = "20221025_M5.1_AlumRock"  # Change this to the desired event ID

def is_clipped(trace, clip_threshold=0.95, window_size=10):
    """
    Check for clipped signals using robust criteria (excluding repeated-value checks).

    Criteria kept:
    - flat peaks/troughs (near-constant high-amplitude windows)
    - asymmetric amplitude distribution (too many extreme positive/negative samples)

    Parameters:
    - clip_threshold: threshold relative to max amplitude to consider for clipping
    - window_size: number of samples to check for flat peaks
    """
    data = trace.data
    abs_data = np.abs(data)
    max_amp = np.max(abs_data)

    # Criterion: Check for flat peaks (zero or near-zero difference between samples)
    for i in range(len(data) - window_size):
        window = data[i:i+window_size]
        if np.abs(window.max() - window.min()) < max_amp * 0.001:  # 0.1% variation threshold
            if np.abs(np.mean(window)) > 0.5 * max_amp:  # Only consider high-amplitude flat regions
                print(f"    {trace.id}: Found flat peak in signal")
                return True

    # Criterion: Check amplitude distribution asymmetry
    pos_peaks = data[data > 0.8 * max_amp]
    neg_peaks = data[data < -0.8 * max_amp]
    if len(pos_peaks) > 100 or len(neg_peaks) > 100:  # Too many extreme values
        print(f"    {trace.id}: Found too many extreme values: +ve={len(pos_peaks)}, -ve={len(neg_peaks)}")
        return True

    return False

def process_traces(
    mseed_file,
    xml_file,
    distance_file,
    max_distance=None,
    output_dir="processed_traces",
    clip_threshold=None,
    zero_frac_threshold=None,
    plot_skipped=False,
    max_plots=10
):
    """
    Process seismic traces with the following steps:
    1. Filter out traces with zeros/NaN/Inf (to avoid issues during response removal)
    2. Remove instrument response
    3. Demean and detrend
    4. Filter by distance
    5. Remove clipped signals using enhanced detection
    6. Save only HN/HL channels

    Parameters:
    - max_distance: maximum hypocentral distance in km (default: from config)
    - clip_threshold: threshold relative to max amplitude (default: from config)
    - zero_frac_threshold: fraction of zero samples allowed (default: from config)
    - plot_skipped: if True, plot the first max_plots skipped traces
    - max_plots: maximum number of skipped traces to plot
    """
    # Use config defaults if not provided
    if max_distance is None:
        max_distance = MAX_DISTANCE_KM
    if clip_threshold is None:
        clip_threshold = CLIP_THRESHOLD
    if zero_frac_threshold is None:
        zero_frac_threshold = ZERO_FRAC_THRESHOLD
    # Read distance information
    df_dist = pd.read_csv(distance_file)
    stations_within_distance = set(
        df_dist[df_dist['hypocentral_distance_km'] <= max_distance]['station']
    )
    
    print(f"Found {len(stations_within_distance)} stations within {max_distance} km")
    
    # Read the data and inventory
    print("Reading mseed and station XML...")
    st = read(mseed_file)
    inv = read_inventory(xml_file)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process traces
    print("Processing traces...")
    st_processed = obspy.Stream()
    skipped_traces = []  # Store info about skipped traces
    plot_count = 0
    
    for tr in st:
        station = tr.stats.station
        channel = tr.stats.channel
        
        # Check if it's HN or HL channel and within distance
        if not (channel.startswith(('HN', 'HL')) and station in stations_within_distance):
            continue
            
        try:
            # Quick data checks on raw trace BEFORE removing instrument response
            data = tr.data
            n_samples = data.size if hasattr(data, 'size') else len(data)
            zero_count = np.count_nonzero(data == 0)
            nan_count = np.count_nonzero(np.isnan(data))
            inf_count = np.count_nonzero(np.isinf(data))

            zero_frac = zero_count / float(n_samples) if n_samples > 0 else 1.0

            # Skip traces that contain NaN or Inf, or have a zero fraction above threshold
            if nan_count > 0 or inf_count > 0 or zero_frac > zero_frac_threshold:
                print(f"Skipping {tr.id}: zeros={zero_count}, nans={nan_count}, infs={inf_count}, zero_frac={zero_frac:.4f}")
                reason = f"zeros={zero_count}, nans={nan_count}, infs={inf_count}, zero_frac={zero_frac:.4f}"
                skipped_traces.append({'trace': tr.copy(), 'reason': reason, 'stage': 'data_quality'})
                continue

            # Create a copy of the trace for processing
            tr_proc = tr.copy()

            # Remove instrument response
            tr_proc.remove_sensitivity(inv)

            # Demean and detrend
            tr_proc.detrend('demean')
            tr_proc.detrend('linear')
            
            # Check for clipping (repeats-based checks removed)
            print(f"Checking {tr_proc.id} for clipping...")
            if is_clipped(tr_proc, clip_threshold=clip_threshold):
                print(f"Skipping clipped trace: {tr_proc.id}")
                skipped_traces.append({'trace': tr_proc.copy(), 'reason': 'clipped', 'stage': 'clipping'})
                continue
            print(f"    {tr_proc.id} passed clipping checks")
            
            # Add to processed stream
            st_processed += tr_proc
            
        except Exception as e:
            print(f"Error processing {tr.id}: {str(e)}")
    
    # Plot skipped traces if requested
    if plot_skipped and len(skipped_traces) > 0:
        plot_dir = os.path.join(output_dir, "skipped_traces_plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        print(f"\nPlotting up to {max_plots} skipped traces...")
        for i, skip_info in enumerate(skipped_traces[:max_plots]):
            tr_skip = skip_info['trace']
            reason = skip_info['reason']
            stage = skip_info['stage']
            
            fig, ax = plt.subplots(figsize=(12, 4))
            times = tr_skip.times()
            ax.plot(times, tr_skip.data, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Skipped: {tr_skip.id}\nReason: {reason}\nStage: {stage}')
            ax.grid(True, alpha=0.3)
            
            plot_file = os.path.join(plot_dir, f"skipped_{i+1:03d}_{tr_skip.id.replace('.', '_')}.png")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"  Saved plot: {plot_file}")
        
        print(f"Plotted {min(len(skipped_traces), max_plots)} out of {len(skipped_traces)} skipped traces")
    
    # Save processed traces
    if len(st_processed) > 0:
        output_file = os.path.join(output_dir, "processed.ms")
        st_processed.write(output_file, format="MSEED")
        print(f"\nSaved {len(st_processed)} processed traces to {output_file}")
        
        # Save list of processed stations
        stations_file = os.path.join(output_dir, "processed_stations.txt")
        with open(stations_file, 'w') as f:
            for tr in st_processed:
                f.write(f"{tr.id}\n")
        print(f"Saved list of processed stations to {stations_file}")
        
        # Save trace metadata with absolute starttime for later use
        import json
        metadata_file = os.path.join(output_dir, "trace_metadata.json")
        metadata = {}
        for tr in st_processed:
            metadata[tr.id] = {
                'starttime': str(tr.stats.starttime),
                'endtime': str(tr.stats.endtime),
                'sampling_rate': float(tr.stats.sampling_rate),
                'npts': int(tr.stats.npts),
                'network': tr.stats.network,
                'station': tr.stats.station,
                'channel': tr.stats.channel
            }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    print("="*80)
    print("TRACE PROCESSING")
    print("="*80)
    print(f"Event ID: {event_id}")
    print()
    
    # Get event paths using config helper
    event_paths = get_event_paths(event_id)
    short_name = event_paths['short_name']
    
    # Construct file paths
    mseed_file = event_paths['mseed']
    xml_file = event_paths['xml']
    distance_file = event_paths['distance_csv']
    output_dir = event_paths['processed_dir']
    
    print(f"File paths:")
    print(f"  MSEED:    {mseed_file}")
    print(f"  XML:      {xml_file}")
    print(f"  Distance: {distance_file}")
    print(f"  Output:   {output_dir}")
    print()
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(mseed_file):
        missing_files.append(f"  ✗ MSEED: {mseed_file}")
    if not os.path.exists(xml_file):
        missing_files.append(f"  ✗ XML: {xml_file}")
    if not os.path.exists(distance_file):
        missing_files.append(f"  ✗ Distance CSV: {distance_file}")
    
    if missing_files:
        print("❌ Missing required files:")
        for msg in missing_files:
            print(msg)
        sys.exit(1)
    
    print("✓ All input files found")
    print()
    
    # Process the traces with plotting of skipped traces
    process_traces(mseed_file, xml_file, distance_file, output_dir=output_dir, 
                   plot_skipped=True, max_plots=10)
    
    print()
    print("="*80)
    print("PROCESSING COMPLETE")
    print(f"Output saved to: {output_dir}")
    print("="*8)
    print()
    
    # Process the traces with plotting of skipped traces
    process_traces(mseed_file, xml_file, distance_file, output_dir=output_dir, 
                   plot_skipped=True, max_plots=10)