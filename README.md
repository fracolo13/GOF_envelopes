# Seismic Envelope Analysis Pipeline

A comprehensive workflow for generating, processing, and comparing synthetic and observed seismic envelopes for earthquake early warning (EEW) applications.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Workflow](#workflow)
5. [Scripts Documentation](#scripts-documentation)
6. [Directory Structure](#directory-structure)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline implements a complete workflow for:

- **Synthetic waveform generation** using finite fault rupture models (HPC/Cluster)
- **Synthetic envelope calculation** from generated waveforms (HPC/Cluster)
- **Observed data processing** from raw miniSEED files (Local)
- **Envelope extraction** with PhaseNet P-wave picking (Local)
- **Comparative analysis** between CUA and standard synthetic envelopes (Local)

The workflow is designed to run on two environments:
- **Cluster (Euler/HPC)**: Scripts 01-02 for synthetic generation
- **Local (MacBook/Desktop)**: Scripts 03-05 for observed data processing and analysis

### Generative Model

The synthetic waveform generation (scripts 01-02) uses the **TQDNE** (Time-dependent Quantized Diffusion Neural Earthquake) model:

🔗 **Repository:** [https://github.com/highfem/tqdne.git](https://github.com/highfem/tqdne.git)

TQDNE is a deep learning model for generating realistic synthetic seismic waveforms based on fault geometry and rupture parameters.

---

## Installation

### Prerequisites

**Cluster Environment (Scripts 01-02):**
```bash
# Python 3.8+
# TQDNE model and dependencies (https://github.com/highfem/tqdne.git)
# ObsPy
# Pandas, NumPy, H5Py

# Install TQDNE
git clone https://github.com/highfem/tqdne.git
cd tqdne
pip install -e .
# Download model weights (follow TQDNE repository instructions)
```

**Local Environment (Scripts 03-05):**
```bash
# Python 3.8+
conda create -n envelope_analysis python=3.9
conda activate envelope_analysis

# Core dependencies
pip install obspy pandas numpy scipy matplotlib seaborn
pip install seisbench  # For PhaseNet picking
pip install scikit-learn
pip install rasterio requests  # For VS30 data handling

# Optional: for Jupyter notebooks
pip install jupyter
```

### Clone and Setup

```bash
# Clone or download the repository
cd /path/to/your/workspace

# Verify configuration
python config.py
```

---

## Configuration

All paths and parameters are centralized in **`config.py`**.

### Key Configuration Steps

1. **Set environment** in `config.py`:
   ```python
   ENVIRONMENT = 'local'  # or 'cluster'
   ```

2. **Local paths** (edit if needed):
   ```python
   BASE_DIR = '/Users/francescoacolosimo/Desktop/SED/envelopes_test'
   ```

3. **Cluster paths** (for scripts 01-02):
   ```python
   CLUSTER_USER = 'fcolosimo'
   CLUSTER_HOME = f'/cluster/home/{CLUSTER_USER}'
   CLUSTER_SCRATCH = f'/cluster/scratch/{CLUSTER_USER}'
   ```

4. **Processing parameters**:
   ```python
   MAX_DISTANCE_KM = 200          # Maximum station distance
   PHASENET_MODEL = "stead"       # PhaseNet variant
   P_WAVE_VELOCITY_KM_S = 6.0     # P-wave velocity for travel time
   VS30_THRESHOLD = 450           # Site classification threshold (m/s)
   ```

### Verify Configuration

```bash
python config.py
```

This will check that all critical paths exist and display the current configuration.

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEISMIC ENVELOPE ANALYSIS PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       CLUSTER/HPC ENVIRONMENT                             │
│                    (Synthetic Generation: 01-02)                          │
└──────────────────────────────────────────────────────────────────────────┘
    ║
    ║  [INPUT: Fault geometry, centroids, rupture parameters]
    ║
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  01_synthetics_generation.py                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Purpose: Generate synthetic waveforms for finite fault rupture          │
│                                                                           │
│  Input:  - Centroids CSV (fault geometry)                                │
│          - TQDNE model checkpoints                                       │
│          - Generative tables                                             │
│                                                                           │
│  Output: - HDF5 waveform files per centroid                              │
│          - Processed MSEED files                                         │
│          - Grouped realizations                                          │
│                                                                           │
│  Runtime: ~2-10 hours (depends on event size)                            │
└─────────────────────────────────────────────────────────────────────────┘
    ║
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  02_euler_synth_envelopes.py                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Purpose: Process HDF5 data and calculate median envelopes               │
│                                                                           │
│  Input:  - HDF5 files from script 01                                     │
│          - Metadata CSV files                                            │
│                                                                           │
│  Process: 1. Extract N, E, Z components                                  │
│           2. Calculate envelopes (max per second)                        │
│           3. Combine horizontal (RMS of N-E)                             │
│           4. Group by magnitude-distance                                 │
│           5. Calculate median across realizations                        │
│                                                                           │
│  Output: - Structured envelope library:                                  │
│            S|R / AZ{azimuth} / {mag} / {dist} /                          │
│              ├── median_envelope_vertical.npy                            │
│              └── median_envelope_horizontal.npy                          │
│                                                                           │
│  Runtime: ~30-60 minutes                                                 │
└─────────────────────────────────────────────────────────────────────────┘
    ║
    ║  [Transfer synthetic envelope library to local machine]
    ║
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         LOCAL ENVIRONMENT                                 │
│              (Observed Data Processing: 03-05)                            │
└──────────────────────────────────────────────────────────────────────────┘
    ║
    ║  [INPUT: Raw earthquake data - .ms, .xml, distance CSV, event JSON]
    ║
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  03_process_obs_mseed.py                                                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Purpose: Process raw seismic traces and prepare for envelope analysis   │
│                                                                           │
│  Input:  - {event_id}.ms (raw miniSEED)                                  │
│          - {event_id}.xml (station inventory)                            │
│          - station_distance_table_{event_id}.csv                         │
│                                                                           │
│  Process: 1. Filter by distance (< MAX_DISTANCE_KM)                      │
│           2. Remove instrument response                                  │
│           3. Demean and detrend                                          │
│           4. Detect and remove clipped signals                           │
│           5. Filter data quality (zeros, NaN, Inf)                       │
│           6. Keep only HN/HL channels                                    │
│                                                                           │
│  Output: - processed.ms                                                  │
│          - processed_stations.txt                                        │
│          - trace_metadata.json (with absolute start times)               │
│          - skipped_traces_plots/ (if plot_skipped=True)                  │
│                                                                           │
│  Runtime: ~5-20 minutes per event                                        │
└─────────────────────────────────────────────────────────────────────────┘
    ║
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  04_create_envelopes_phasenet.py                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Purpose: Extract envelopes with PhaseNet P-wave picking                 │
│                                                                           │
│  Input:  - processed.ms (from script 03)                                 │
│          - station_distance_table_{event_id}.csv                         │
│          - {event_id}.json (event origin, depth)                         │
│                                                                           │
│  Process: 1. Load 3-component waveforms per station                      │
│           2. Pick P-wave using PhaseNet (TauP-constrained)               │
│           3. Fallback to AIC picker if PhaseNet fails                    │
│           4. Calculate envelopes (max per second):                       │
│              - Vertical: Z component                                     │
│              - Horizontal: RMS(N, E)                                     │
│           5. Store P-arrival metadata                                    │
│                                                                           │
│  Output: - {NETWORK.STATION}/                                            │
│              ├── Vertical/                                               │
│              │   ├── envelope_HNZ.npy (or HLZ)                           │
│              │   └── envelope_HNZ_meta.json                              │
│              └── Horizontal_combined/                                    │
│                  ├── envelope_HNE_HNN.npy (or HLE_HLN)                   │
│                  └── envelope_HNE_HNN_meta.json                          │
│          - station_picks.csv (picking statistics)                        │
│          - pick_qc_plots.png (visual QC)                                 │
│                                                                           │
│  Runtime: ~10-30 minutes per event                                       │
└─────────────────────────────────────────────────────────────────────────┘
    ║
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  03_combined_0_5CUA_synthetic_heatmap_comparison.py                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Purpose: Compare CUA and standard synthetic envelopes against observed  │
│                                                                           │
│  Input:  - Observed envelopes (from script 04)                           │
│          - CUA synthetic templates (CUA_H.npy, CUA_Z.npy)                │
│          - Standard synthetic templates (ML_H.npy, ML_Z.npy)             │
│          - VS30 data for site classification                             │
│                                                                           │
│  Process: 1. Load observed and synthetic envelopes                       │
│           2. For each wrong-source parameter combination:                │
│              - Magnitude errors: ±0.5 in 0.1 steps                       │
│              - Distance offsets: 0-130 km in 10 km steps                 │
│              - Random epicenter shifts                                   │
│           3. Calculate metrics:                                          │
│              - Goodness-of-fit (GoF)                                     │
│              - Correlation coefficient                                   │
│              - Peak amplitude ratio                                      │
│           4. Generate side-by-side heatmaps (CUA vs Synthetic)           │
│                                                                           │
│  Output: - High-resolution heatmaps (300 dpi):                           │
│              {event}_gof_combined_comparison.png                         │
│              {event}_correlation_combined_comparison.png                 │
│              {event}_peak_ratio_combined_comparison.png                  │
│          - Station count heatmaps                                        │
│          - Summary statistics CSV                                        │
│                                                                           │
│  Runtime: ~30-120 minutes per event (depends on N_RANDOM_TRIES)          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Scripts Documentation

### 01_synthetics_generation.py

**Environment:** Cluster/HPC (Euler)

**Purpose:** Generate synthetic waveforms for finite fault rupture using TQDNE model

**Configuration in script:**
```python
eq = "Kumamoto_6"                    # Event identifier
date = "2019-07-06T03:19:53"         # Event origin time
```

**Key features:**
- Iterates over fault centroids from CSV
- Generates waveforms using TQDNE neural network model
- Writes metadata to seisbench format
- Processes waveforms through 4-step pipeline:
  1. Write to MSEED format
  2. Adapt frequency content
  3. Group realizations by station
  4. Process frequency characteristics

**Output:**
```
{synthetics_dir}/
├── {timestamp}_synthetics/
│   ├── mseed_raw/                   # Raw synthetic MSEED files
│   ├── processed_mseeds/
│   │   ├── realizations/            # Individual realizations
│   │   └── summed/                  # Summed/averaged realizations
│   └── grouped_centroids_data_with_magnitude.csv
```

**Usage:**
```bash
# On cluster
python 01_synthetics_generation.py
```

---

### 02_euler_synth_envelopes.py

**Environment:** Cluster/HPC (Euler)

**Purpose:** Process HDF5 waveform data and calculate median envelopes

**Key features:**
- Reads HDF5 files with waveform data (N, E, Z components)
- Calculates envelopes as max absolute value per second
- Combines horizontal components using RMS: `sqrt((N² + E²) / 2)`
- Groups by magnitude and distance
- Calculates median across multiple realizations
- Supports azimuth-specific templates (AZ80, AZ90, etc.)

**Output structure:**
```
output/synthetic_envelopes/
├── S/                               # Soft soil sites (VS30 < 450 m/s)
│   └── AZ80/                        # Azimuth tag
│       └── 6.0/                     # Magnitude
│           └── 25/                  # Distance (km)
│               ├── median_envelope_vertical.npy
│               └── median_envelope_horizontal.npy
└── R/                               # Rock sites (VS30 >= 450 m/s)
    └── AZ80/
        └── ...
```

**Usage:**
```bash
# On cluster
python 02_euler_synth_envelopes.py
```

---

### 03_process_obs_mseed.py

**Environment:** Local

**Purpose:** Process raw observed seismic data

**Configuration in script:**
```python
event_id = "20221025_M5.1_AlumRock"  # Edit for each event
```

**Key features:**
- **Quality control filters:**
  - Removes traces with >50% zeros
  - Removes traces with NaN or Inf values
  - Detects and removes clipped signals
- **Processing steps:**
  - Removes instrument response using station XML
  - Demeans and detrends
  - Filters by distance (configurable in config.py)
  - Keeps only HN/HL channels
- **Optional diagnostics:** Plots skipped traces for QC

**Clipping detection criteria:**
- Flat peaks (near-constant high-amplitude windows)
- Asymmetric amplitude distribution

**Output:**
```
processed_traces_{event}/
├── processed.ms                     # Processed miniSEED
├── processed_stations.txt           # List of stations
├── trace_metadata.json              # Absolute timing info
└── skipped_traces_plots/            # QC plots (optional)
    └── skipped_001_CI_ASCO_HNE.png
```

**Usage:**
```bash
python 03_process_obs_mseed.py
```

**Input files required:**
- `{event_id}.ms` - Raw miniSEED file
- `{event_id}.xml` - Station inventory XML
- `station_distance_table_{event_id}.csv` - Distance information

---

### 04_create_envelopes_phasenet.py

**Environment:** Local

**Purpose:** Extract envelopes with PhaseNet P-wave picking

**Configuration in script:**
```python
event_id = "20221025_M5.1_AlumRock"        # Edit for each event
PHASENET_MODEL_OVERRIDE = None             # Or "instance", "geofon", "scedc"
```

**Key features:**
- **P-wave picking strategy:**
  1. PhaseNet neural network picker (primary)
  2. TauP travel-time constraint (±10s window)
  3. AIC picker fallback if PhaseNet fails
  4. Physical validation (travel time > 0, < 200s)

- **Envelope calculation:**
  - Per-second sampling (max absolute value)
  - Vertical: Z component
  - Horizontal: RMS combination of N-E components

- **Metadata storage:**
  - P-arrival index in envelope
  - Event origin time
  - P-wave travel time
  - Absolute start times

**PhaseNet models available:**
- `stead`: STEAD dataset (recommended for general use)
- `instance`: INSTANCE dataset (M5-7.5 range)
- `original`: Original PhaseNet
- `geofon`: Global GEOFON data
- `scedc`: Southern California

**Output:**
```
envelopes_{event}/
├── CI.ASCO/
│   ├── Vertical/
│   │   ├── envelope_HNZ.npy
│   │   └── envelope_HNZ_meta.json
│   └── Horizontal_combined/
│       ├── envelope_HNE_HNN.npy
│       └── envelope_HNE_HNN_meta.json
├── station_picks.csv                # Picking statistics
└── pick_qc_plots.png                # Visual QC
```

**Usage:**
```bash
python 04_create_envelopes_phasenet.py

# With options
python 04_create_envelopes_phasenet.py --limit-stations 10  # Test on 10 stations
```

**Picking success statistics:**
- Reports picks by method (PhaseNet, AIC, fallback)
- Shows confidence distributions
- Generates QC plots for visual inspection

---

### 03_combined_0_5CUA_synthetic_heatmap_comparison.py

**Environment:** Local

**Purpose:** Compare CUA and standard synthetic envelopes against observed data

**Configuration in script:**
```python
event_ids = ["20190705_M7.1_Ridgecrest", "20140824_M6.0_SouthNapa", ...]
```

**Key features:**
- **Equal comparison framework:**
  - Same random epicenter shifts for both methods
  - Same magnitude errors and distance offsets
  - Same time windows and metrics

- **Wrong-source parameters tested:**
  - Magnitude errors: -0.5 to +0.5 (0.1 steps)
  - Distance offsets: 0 to 130 km (10 km steps)
  - Random epicenter shifts (100 realizations per offset)

- **Metrics calculated:**
  - **GoF (Goodness-of-Fit):** `sqrt(amplitude_fit × correlation)`
  - **Correlation:** Pearson correlation coefficient
  - **Peak ratio:** `obs_max / syn_max`

- **Progressive time windows:**
  - Analyzes multiple windows: 4, 5, 6, 8, 10, 15, 20, 30 seconds
  - Only includes stations where P-wave has arrived
  - Minimum 2 seconds of data required after P-arrival

- **Station filtering:**
  - Only counts pairs where at least one envelope has signal
  - Excludes noise-vs-noise comparisons

**Output:**
```
results/{event}/
├── heatmaps/
│   ├── {event}_gof_combined_comparison_tw{X}s.png
│   ├── {event}_correlation_combined_comparison_tw{X}s.png
│   └── {event}_peak_ratio_combined_comparison_tw{X}s.png
├── statistics/
│   ├── station_counts_tw{X}s.csv
│   └── metrics_summary.csv
└── plots/
    └── ...
```

**Heatmap interpretation:**
- **X-axis:** Distance offset (0-130 km)
- **Y-axis:** Magnitude error (-0.5 to +0.5)
- **Color:** Metric value (GoF, correlation, or peak ratio)
- **Side-by-side:** Left = CUA, Right = Standard synthetic
- **White cells:** No data or insufficient stations

**Usage:**
```bash
python 03_combined_0_5CUA_synthetic_heatmap_comparison.py
```

**Runtime optimization:**
- Adjust `N_RANDOM_TRIES` in config.py (default: 100)
- Process fewer events to reduce time
- Use parallel processing (future enhancement)

---

## Directory Structure

### Expected Data Organization

```
envelopes_test/
├── codes/
│   └── repo_ready/
│       ├── config.py                    # ← Global configuration
│       ├── README.md                    # ← This file
│       ├── 01_synthetics_generation.py
│       ├── 02_euler_synth_envelopes.py
│       ├── 03_process_obs_mseed.py
│       ├── 04_create_envelopes_phasenet.py
│       └── 03_combined_0_5CUA_synthetic_heatmap_comparison.py
│
├── data/
│   ├── maren_eq/                        # Raw earthquake data
│   │   ├── {event_id}.ms
│   │   ├── {event_id}.xml
│   │   ├── {event_id}.json
│   │   └── station_distance_table_{event_id}.csv
│   │
│   ├── operational_processed/           # Processed traces (script 03)
│   │   └── processed_traces_{event}/
│   │
│   ├── operational_envelopes/           # Observed envelopes (script 04)
│   │   └── envelopes_{event}/
│   │
│   ├── aligned_envelopes_improved/      # CUA templates
│   │   └── {mag}/{dist}/
│   │       ├── CUA_H.npy
│   │       └── CUA_Z.npy
│   │
│   ├── synthetic_4_8/                   # Standard synthetic templates
│   │   └── R|S/{mag}/{dist}/
│   │       ├── ML_H.npy
│   │       └── ML_Z.npy
│   │
│   └── vs30/
│       ├── vs30_stations.csv
│       └── vs30.tif
│
└── results/
    └── operational_results/             # Final outputs (script 05)
        └── {event}/
            ├── heatmaps/
            ├── statistics/
            └── plots/
```

### Input Data Requirements

**For each event, you need:**

1. **Raw waveform data:**
   - `{event_id}.ms` - miniSEED file with raw traces
   - `{event_id}.xml` - StationXML with response information

2. **Metadata:**
   - `{event_id}.json` - Event origin time, location, depth
   - `station_distance_table_{event_id}.csv` - Station distances

3. **Synthetic templates:**
   - CUA library: `aligned_envelopes_improved/`
   - Standard library: `synthetic_4_8/`

4. **Site characterization:**
   - `vs30_stations.csv` - VS30 values for known stations
   - `vs30.tif` (optional) - VS30 raster for spatial lookup

---

## Usage Examples

### Example 1: Complete Workflow for One Event

```bash
# 1. Configure
cd /path/to/envelopes_test/codes/repo_ready
python config.py  # Verify paths

# 2. Edit event ID in script 03
# In 03_process_obs_mseed.py, set:
# event_id = "20140824_M6.0_SouthNapa"

# 3. Process observed data
python 03_process_obs_mseed.py

# 4. Edit event ID in script 04
# In 04_create_envelopes_phasenet.py, set:
# event_id = "20140824_M6.0_SouthNapa"

# 5. Create envelopes
python 04_create_envelopes_phasenet.py

# 6. Edit event list in script 05
# In 03_combined_0_5CUA_synthetic_heatmap_comparison.py, set:
# event_ids = ["20140824_M6.0_SouthNapa"]

# 7. Run comparison
python 03_combined_0_5CUA_synthetic_heatmap_comparison.py
```

### Example 2: Batch Processing Multiple Events

```bash
# Create a batch script
cat > process_events.sh << 'EOF'
#!/bin/bash

events=(
    "20140824_M6.0_SouthNapa"
    "20190705_M7.1_Ridgecrest"
    "20240403_M7.4_Hualien"
)

for event in "${events[@]}"; do
    echo "Processing $event..."
    
    # Update event_id in scripts (requires sed or manual edit)
    sed -i.bak "s/event_id = .*/event_id = \"$event\"/" 03_process_obs_mseed.py
    sed -i.bak "s/event_id = .*/event_id = \"$event\"/" 04_create_envelopes_phasenet.py
    
    # Run processing
    python 03_process_obs_mseed.py
    python 04_create_envelopes_phasenet.py
done

# Run comparison for all events
python 03_combined_0_5CUA_synthetic_heatmap_comparison.py
EOF

chmod +x process_events.sh
./process_events.sh
```

### Example 3: Quick Test on Limited Stations

```bash
# Test envelope creation on 5 stations only
python 04_create_envelopes_phasenet.py --limit-stations 5

# Check output
ls data/operational_envelopes/envelopes_south_napa/
```

### Example 4: Cluster Workflow (Scripts 01-02)

```bash
# On cluster (Euler)
# 1. Set environment in config.py
# ENVIRONMENT = 'cluster'

# 2. Edit event in script 01
# eq = "Ridgecrest_7"
# date = "2019-07-06T03:19:53"

# 3. Submit job (example SLURM)
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=synth_gen
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4

module load python/3.9
conda activate tqdne_env

python 01_synthetics_generation.py
EOF

# 4. After completion, process envelopes
python 02_euler_synth_envelopes.py

# 5. Transfer results to local machine
rsync -avz euler:/cluster/scratch/fcolosimo/Data/envelopes/output/ \
    /Users/.../envelopes_test/data/synthetic_4_8/
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: Cannot find config.py

**Problem:**
```
ModuleNotFoundError: No module named 'config'
```

**Solution:**
```bash
# Make sure you're in the correct directory
cd /path/to/envelopes_test/codes/repo_ready

# Check config.py exists
ls config.py

# Run scripts from this directory
python 03_process_obs_mseed.py
```

#### 2. Missing Input Files

**Problem:**
```
❌ MSEED file not found: /path/to/file.ms
```

**Solution:**
- Verify `DATA_DIR` in config.py points to correct location
- Check file naming matches expected format: `{event_id}.ms`
- Use `get_event_paths()` to debug:
```python
from config import get_event_paths
paths = get_event_paths("20140824_M6.0_SouthNapa")
print(paths)
```

#### 3. PhaseNet Model Download Failed

**Problem:**
```
⚠️  Failed to load PhaseNet model 'stead': ...
```

**Solution:**
```bash
# Manually download model
python -c "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('stead')"

# Or try different model in config.py
PHASENET_MODEL = "instance"  # or "original", "geofon"
```

#### 4. VS30 Data Not Found

**Problem:**
```
Warning: VS30 CSV not found
```

**Solution:**
- Download VS30 station data from [ESM database](https://esm-db.eu)
- Place CSV in `data/vs30/vs30_stations.csv`
- Script will fall back to ESM web service if local data missing

#### 5. Clipped Traces Detected

**Problem:**
Too many traces are being flagged as clipped

**Solution:**
- Adjust `CLIP_THRESHOLD` in config.py (default: 0.95)
- Check clipping plots in `skipped_traces_plots/`
- Consider lowering `ZERO_FRAC_THRESHOLD` if many false positives

#### 6. Low Picking Success Rate

**Problem:**
PhaseNet picking success < 50%

**Solution:**
```python
# Try different PhaseNet model
PHASENET_MODEL = "instance"  # Good for M5-7.5 range
# or
PHASENET_MODEL = "geofon"    # Good for global events

# Check picking statistics in station_picks.csv
# Review pick_qc_plots.png for visual inspection
```

#### 7. Empty Heatmaps / All NaN Values

**Problem:**
Heatmaps show no data (all white)

**Solution:**
- Check that envelope files exist for both observed and synthetic
- Verify VS30 site classification is matching stations correctly
- Try shorter time windows (4-6 seconds)
- Check distance range includes your stations
- Ensure `N_RANDOM_TRIES` > 0 in config.py

#### 8. Memory Issues with Large Events

**Problem:**
Script crashes with MemoryError

**Solution:**
```python
# Process fewer stations at a time
python 04_create_envelopes_phasenet.py --limit-stations 50

# Reduce N_RANDOM_TRIES in config.py
N_RANDOM_TRIES = 50  # instead of 100

# Or run on machine with more RAM
```

### Performance Optimization

**Script 01-02 (Cluster):**
- Use batch job submission for parallel processing
- Request sufficient memory (8-16 GB per job)
- Monitor I/O - use scratch space for temp files

**Script 03 (Processing):**
- Typical: 5-20 minutes per event
- Bottleneck: Instrument response removal
- Can process multiple events in parallel

**Script 04 (Envelopes):**
- Typical: 10-30 minutes per event
- Bottleneck: PhaseNet inference on CPU
- Use GPU if available (requires seisbench with CUDA)

**Script 05 (Comparison):**
- Typical: 30-120 minutes per event
- Bottleneck: N_RANDOM_TRIES × stations × cells
- Reduce N_RANDOM_TRIES for faster testing
- Process fewer events in batch

### Getting Help

For issues not covered here:

1. Check error messages carefully - they often indicate missing files or paths
2. Verify configuration with `python config.py`
3. Review intermediate outputs (CSV files, plots)
4. Check data formats match expected structure
5. Contact: francesco.colosimo@sed.ethz.ch

---

## Advanced Configuration

### Custom PhaseNet Models

Train custom PhaseNet model on your data:
```bash
# See seisbench documentation
# https://github.com/seisbench/seisbench
```

### Custom Velocity Models

Edit config.py to use different P-wave velocity:
```python
P_WAVE_VELOCITY_KM_S = 5.5  # Slower crustal velocity
# or
P_WAVE_VELOCITY_KM_S = 8.0  # Upper mantle velocity
```

### Custom Distance Ranges

```python
# In config.py
DISTANCE_OFFSETS_KM = list(range(0, 201, 20))  # 0-200 km in 20 km steps
MAX_DISTANCE_KM = 300  # Include more distant stations
```

### Custom Magnitude Ranges

```python
# In config.py
MAGNITUDE_ERRORS = np.arange(-1.0, 1.1, 0.2)  # Larger range, coarser steps
```

---

## Citation

If you use this pipeline in your research, please cite:

```
Colosimo, F. (2025). Seismic Envelope Analysis Pipeline for Earthquake Early Warning.
Master's Thesis, ETH Zurich / SED.
```

---

## License

This code is provided for research purposes. Please contact the author for commercial use.

---

## Changelog

### Version 1.0 (2025-05-27)
- Initial release with centralized configuration
- Complete workflow documentation
- Support for both cluster and local environments
- PhaseNet integration for P-wave picking
- Side-by-side CUA vs synthetic comparison

---

**Last Updated:** May 27, 2025  
**Author:** Francesco Colosimo  
**Contact:** francesco.colosimo@sed.ethz.ch  
**Institution:** Swiss Seismological Service (SED), ETH Zurich
