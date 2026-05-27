#!/usr/bin/env python3
"""
Global Configuration for Seismic Envelope Analysis Pipeline
============================================================

This file centralizes all path configurations and parameters used across
the envelope analysis workflow. Edit these paths to match your local setup
or cluster environment.

Directory Structure Expected:
-----------------------------
DATA_DIR/
├── maren_eq/                      # Raw earthquake data
│   ├── <event_id>.ms             # Raw miniSEED files
│   ├── <event_id>.xml            # Station inventory XML
│   └── station_distance_table_<event_id>.csv
│
├── operational_processed/         # Processed traces (output of 03_process_obs_mseed.py)
│   └── processed_traces_<event>/
│       ├── processed.ms
│       ├── processed_stations.txt
│       └── trace_metadata.json
│
├── operational_envelopes/         # Observed envelopes (output of 04_create_envelopes_phasenet.py)
│   └── <NETWORK.STATION>/
│       ├── Vertical/envelope.npy
│       └── Horizontal_combined/envelope.npy
│
├── aligned_envelopes_improved/    # CUA synthetic envelope templates
│   └── <mag>/<dist>/
│       ├── CUA_H.npy
│       └── CUA_Z.npy
│
├── synthetic_4_8/                 # Standard synthetic envelope templates
│   └── R|S/
│       └── <mag>/<dist>/
│           ├── ML_H.npy
│           └── ML_Z.npy
│
└── vs30/
    ├── vs30_stations.csv
    └── vs30.tif

RESULTS_BASE/
├── operational_results/           # Final comparison results
│   └── <event>/
│       ├── heatmaps/
│       ├── statistics/
│       └── plots/

For Cluster/HPC Usage:
----------------------
On Euler or other HPC systems, modify CLUSTER_* paths for the synthetic
generation workflow (scripts 01 and 02).
"""

import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT SELECTION
# ============================================================================
# Set to 'local' for local machine or 'cluster' for HPC environment
ENVIRONMENT = 'local'  # Options: 'local', 'cluster'

# ============================================================================
# LOCAL PATHS (MacBook / Desktop)
# ============================================================================
if ENVIRONMENT == 'local':
    # Base directories
    BASE_DIR = '/Users/francescoacolosimo/Desktop/SED/envelopes_test'
    DATA_BASE = os.path.join(BASE_DIR, 'data')
    RESULTS_BASE = os.path.join(BASE_DIR, 'results')
    
    # Input data paths
    DATA_DIR = os.path.join(DATA_BASE, 'maren_eq')           # Raw earthquake data (.ms, .xml, distance CSVs)
    
    # Processing output paths
    PROCESSED_DIR = os.path.join(DATA_BASE, 'operational_processed')  # Output of 03_process_obs_mseed.py
    ENVELOPES_DIR = os.path.join(DATA_BASE, 'operational_envelopes')  # Output of 04_create_envelopes_phasenet.py
    
    # Template library paths
    CUA_BASE = os.path.join(DATA_BASE, 'aligned_envelopes_improved')  # CUA synthetic templates
    SYNTHETIC_BASE = os.path.join(DATA_BASE, 'synthetic_4_8')         # Standard synthetic templates
    SYN_CUA_ENV = os.path.join(DATA_BASE, 'syn_cua_env')              # Unified template library
    
    # VS30 data paths
    VS30_CSV = os.path.join(DATA_BASE, 'vs30', 'vs30_stations.csv')
    VS30_TIFF = os.path.join(DATA_BASE, 'vs30', 'vs30.tif')
    
    # Results paths
    RESULTS_DIR = os.path.join(RESULTS_BASE, 'operational_results')

# ============================================================================
# CLUSTER PATHS (Euler/HPC)
# ============================================================================
elif ENVIRONMENT == 'cluster':
    # User-specific cluster paths
    CLUSTER_USER = 'fcolosimo'
    
    # Base directories
    CLUSTER_HOME = f'/cluster/home/{CLUSTER_USER}'
    CLUSTER_SCRATCH = f'/cluster/scratch/{CLUSTER_USER}'
    
    # Data paths
    CLUSTER_DATA_BASE = os.path.join(CLUSTER_SCRATCH, 'Data')
    CLUSTER_CODES_BASE = os.path.join(CLUSTER_HOME, 'Codes')
    
    # For synthetic generation (scripts 01 & 02)
    SYNTHETIC_EVENT_DIR = None  # Set dynamically per event (e.g., Kumamoto_6)
    SYNTHETIC_OUTPUT_BASE = CLUSTER_SCRATCH  # Base for synthetic outputs
    
    # Code paths for synthetic generation
    TQDNE_CHECKPOINT_EDM = os.path.join(CLUSTER_CODES_BASE, 'tqdne/weights/edm.ckpt')
    TQDNE_CHECKPOINT_AUTOENCODER = os.path.join(CLUSTER_CODES_BASE, 'tqdne/weights/autoencoder.ckpt')
    TQDNE_WRITE_SCRIPT = os.path.join(CLUSTER_CODES_BASE, 'tqdne/scripts/write_to_seisbench.py')
    
    # Processing scripts (for synthetic generation)
    PROCESSING_SCRIPTS = {
        'write_mseed': os.path.join(CLUSTER_CODES_BASE, 'Finite_rupture/processing/01_writing_to_mseed.py'),
        'adapt_frequency': os.path.join(CLUSTER_CODES_BASE, 'Finite_rupture/processing/02_adapting_frequency.py'),
        'group_realizations': os.path.join(CLUSTER_CODES_BASE, 'Finite_rupture/processing/04_group_realisations.py'),
        'process_frequency': os.path.join(CLUSTER_CODES_BASE, 'Finite_rupture/processing/05_frequency.py'),
    }
    
    # Envelope processing paths
    ENVELOPES_DATA_DIR = os.path.join(CLUSTER_SCRATCH, 'Data/envelopes')
    ENVELOPES_OUTPUT_DIR = os.path.join(CLUSTER_SCRATCH, 'Data/envelopes/output/synthetic_envelopes')

else:
    raise ValueError(f"Invalid ENVIRONMENT: {ENVIRONMENT}. Must be 'local' or 'cluster'")

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Distance filtering
MAX_DISTANCE_KM = 200  # Maximum hypocentral distance for station filtering

# Clipping detection
CLIP_THRESHOLD = 0.95          # Threshold for clipping detection (95% of max amplitude)
ZERO_FRAC_THRESHOLD = 0.50     # Maximum fraction of zero samples allowed (50%)

# PhaseNet model selection
# Options: "instance", "original", "geofon", "scedc", "stead"
PHASENET_MODEL = "stead"

# P-wave velocity for travel time calculations
P_WAVE_VELOCITY_KM_S = 6.0     # km/s (typical crustal velocity)

# Envelope processing
SAMPLING_RATE = 100            # Hz - standard sampling rate for processing
ENVELOPE_WINDOW_S = 60         # seconds - window length for envelope analysis

# Heatmap comparison parameters
TIME_WINDOWS_S = [4, 5, 6, 8, 10, 15, 20, 30]  # Time windows for progressive analysis
N_RANDOM_TRIES = 100           # Number of random epicenter shifts per distance offset
MAGNITUDE_ERRORS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DISTANCE_OFFSETS_KM = list(range(0, 131, 10))  # 0 to 130 km in 10 km steps

# VS30 site classification threshold
VS30_THRESHOLD = 450           # m/s - below = soft soil (S), above = rock (R)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_event_paths(event_id):
    """
    Generate all relevant file paths for a given event ID.
    
    Parameters:
    -----------
    event_id : str
        Event identifier (e.g., "20140824_M6.0_SouthNapa")
    
    Returns:
    --------
    dict
        Dictionary containing all relevant paths for the event
    """
    # Extract short name from event_id (e.g., "SouthNapa" -> "south_napa")
    import re
    event_name = event_id.split('_')[-1]
    short_name = re.sub(r'(?<!^)(?=[A-Z])', '_', event_name).lower()
    
    paths = {
        'event_id': event_id,
        'short_name': short_name,
        
        # Input files
        'mseed': os.path.join(DATA_DIR, f"{event_id}.ms"),
        'xml': os.path.join(DATA_DIR, f"{event_id}.xml"),
        'distance_csv': os.path.join(DATA_DIR, f"station_distance_table_{event_id}.csv"),
        'event_info': os.path.join(DATA_DIR, f"{event_id}.json"),
        
        # Processing outputs
        'processed_dir': os.path.join(PROCESSED_DIR, f"processed_traces_{short_name}"),
        'envelopes_dir': os.path.join(ENVELOPES_DIR, f"envelopes_{short_name}"),
        
        # Results
        'results_dir': os.path.join(RESULTS_DIR, short_name),
    }
    
    return paths


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    if ENVIRONMENT == 'local':
        dirs = [
            DATA_BASE,
            DATA_DIR,
            PROCESSED_DIR,
            ENVELOPES_DIR,
            RESULTS_DIR,
        ]
    else:  # cluster
        dirs = [
            ENVELOPES_DATA_DIR,
            ENVELOPES_OUTPUT_DIR,
        ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Verified/created all required directories for {ENVIRONMENT} environment")


def validate_paths():
    """Validate that critical paths exist."""
    if ENVIRONMENT == 'local':
        critical_paths = [
            (DATA_DIR, "Raw data directory"),
            (VS30_CSV, "VS30 station data"),
        ]
    else:  # cluster
        critical_paths = [
            (CLUSTER_DATA_BASE, "Cluster data directory"),
            (TQDNE_CHECKPOINT_EDM, "TQDNE EDM checkpoint"),
            (TQDNE_CHECKPOINT_AUTOENCODER, "TQDNE autoencoder checkpoint"),
        ]
    
    missing = []
    for path, description in critical_paths:
        if not os.path.exists(path):
            missing.append(f"  ✗ {description}: {path}")
    
    if missing:
        print("⚠️  Warning: Some paths are missing:")
        for msg in missing:
            print(msg)
        return False
    else:
        print(f"✓ All critical paths validated for {ENVIRONMENT} environment")
        return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SEISMIC ENVELOPE ANALYSIS - CONFIGURATION")
    print("="*80)
    print(f"Environment: {ENVIRONMENT}")
    print()
    
    if ENVIRONMENT == 'local':
        print("Local Paths:")
        print(f"  Base directory:    {BASE_DIR}")
        print(f"  Data directory:    {DATA_DIR}")
        print(f"  Results directory: {RESULTS_DIR}")
    else:
        print("Cluster Paths:")
        print(f"  Home:    {CLUSTER_HOME}")
        print(f"  Scratch: {CLUSTER_SCRATCH}")
    
    print()
    ensure_directories()
    validate_paths()
    
    print()
    print("Configuration loaded successfully!")
    print("="*80)
