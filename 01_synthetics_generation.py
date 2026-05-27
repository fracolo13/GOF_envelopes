#!/usr/bin/env python3
"""
Synthetic Waveform Generation for Finite Fault Rupture
========================================================

This script generates synthetic waveforms for finite fault rupture scenarios using
the TQDNE (Time-dependent Quantized Diffusion Neural Earthquake) model.

Prerequisites:
- Centroids CSV file with fault geometry and rupture parameters
- TQDNE model checkpoints (EDM and autoencoder)
- Generative tables for each centroid

Workflow:
1. Read centroids data (lat, lon, depth, magnitude, rupture time)
2. Generate synthetic waveforms for each centroid using TQDNE
3. Write metadata and convert to seisbench format
4. Process waveforms (write to MSEED, adapt frequency, group realizations)

Output:
- HDF5 files with synthetic waveforms
- Processed MSEED files
- Grouped realizations by station

Usage:
    python 01_synthetics_generation.py
    
Note: This script is designed to run on HPC clusters (Euler).
      Edit config.py to set ENVIRONMENT='cluster' and adjust paths.
"""

import os
import pandas as pd
import time
from datetime import datetime
import sys

# Import configuration
from config import (
    ENVIRONMENT,
    CLUSTER_DATA_BASE,
    SYNTHETIC_OUTPUT_BASE,
    TQDNE_CHECKPOINT_EDM,
    TQDNE_CHECKPOINT_AUTOENCODER,
    TQDNE_WRITE_SCRIPT,
    PROCESSING_SCRIPTS,
)

# Ensure we're running in cluster environment
if ENVIRONMENT != 'cluster':
    print("⚠️  Warning: This script is designed for cluster environment.")
    print("   Set ENVIRONMENT='cluster' in config.py or adjust paths manually.")

# ============================================================================
# EVENT CONFIGURATION - Edit these for each event
# ============================================================================
eq = "Kumamoto_6"
# date = "2016-04-15T16:25:06"
date = "2019-07-06T03:19:53"

base_path = os.path.join(CLUSTER_DATA_BASE, eq)
out_path = os.path.join(SYNTHETIC_OUTPUT_BASE, 'Data', eq)

# Load the centroids data
centroids_df = pd.read_csv(f'{base_path}/fault_csv/grouped_centroids_data_with_magnitude.csv')

# Create a directory with the current UTC time and _synthetics
current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
synthetics_dir = os.path.join(out_path, f'{current_time}_synthetics')
os.makedirs(synthetics_dir, exist_ok=True)



# Create additional directories inside synthetics_dir
os.makedirs(os.path.join(synthetics_dir, 'mseed_raw'), exist_ok=True)
os.makedirs(os.path.join(synthetics_dir, 'processed_mseeds/realizations'), exist_ok=True)
os.makedirs(os.path.join(synthetics_dir, 'processed_mseeds/summed'), exist_ok=True)


# Save the centroids data to the synthetics directory
centroids_df.to_csv(os.path.join(synthetics_dir, 'grouped_centroids_data_with_magnitude.csv'), index=False)

# Iterate over each centroid
for centroid in centroids_df.to_dict(orient='records'):
    centroid_lat = centroid['centroid_lat']
    centroid_lon = centroid['centroid_lon']
    centroid_depth = centroid['centroid_depth']
    trup = centroid['trup']
    magnitude = centroid['magnitude']
    csv_file = f"{base_path}/generative/generative_tables/generative_centroid_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv"
    output_hdf5 = f"{synthetics_dir}/centroid_{centroid_lat}_{centroid_lon}_{trup}_{magnitude}.hdf5"
    
    # Estimate time remaining
    start_time = time.time()

    # Command to produce synthetics using TQDNE model
    command1 = f"generate-waveforms --outfile {output_hdf5} --edm_checkpoint {TQDNE_CHECKPOINT_EDM} --autoencoder_checkpoint {TQDNE_CHECKPOINT_AUTOENCODER} --csv {base_path}/generative/generative_tables/generative_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv"
    os.system(command1)
    
    # Command to write metadata and convert to seisbench format
    command2 = f"python {TQDNE_WRITE_SCRIPT} {base_path}/generative/centroid_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv {output_hdf5} {synthetics_dir}"
    os.system(command2)

    current_index = centroids_df.index[centroids_df['centroid_lat'] == centroid_lat].tolist()[0]
    total = len(centroids_df)
    print(f'Processing {current_index + 1} out of {total} centroids')
    
    
    elapsed_time = time.time() - start_time
    eta = elapsed_time * (total - (current_index + 1))
    if eta > 3600:
        eta_hours = eta / 3600
        print(f'Estimated time remaining: {eta_hours:.2f} hours')
    elif eta > 60:
        eta_minutes = eta / 60
        print(f'Estimated time remaining: {eta_minutes:.2f} minutes')
    else:
        print(f'Estimated time remaining: {eta:.2f} seconds')

# Post-processing commands using configured script paths
print("\n" + "="*80)
print("STARTING POST-PROCESSING")
print("="*80)

print("\n[1/4] Writing to MSEED format...")
command3 = f"python {PROCESSING_SCRIPTS['write_mseed']} {synthetics_dir} {eq} {date}"
os.system(command3)

print("\n[2/4] Adapting frequency...")
command4 = f"python {PROCESSING_SCRIPTS['adapt_frequency']} {synthetics_dir} {eq} {date}"
os.system(command4)

print("\n[3/4] Grouping realizations...")
command5 = f"python {PROCESSING_SCRIPTS['group_realizations']} {synthetics_dir} {eq} {date}"
os.system(command5)

print("\n[4/4] Processing frequency...")
command6 = f"python {PROCESSING_SCRIPTS['process_frequency']} {synthetics_dir} {eq} {date}"
os.system(command6)

print("\n" + "="*80)
print("POST-PROCESSING COMPLETE")
print(f"Output directory: {synthetics_dir}")
print("="*80)
    


