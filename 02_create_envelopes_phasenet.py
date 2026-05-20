
#!/usr/bin/env python3

# ============================================================================
# CONFIGURATION  — edit these before running
# ============================================================================
event_id       = "20221025_M5.1_AlumRock"  # Change this to the desired event ID

# PhaseNet model selection
# Available models: "instance", "original", "geofon", "scedc"
# - instance: INSTANCE dataset (diverse, good for M5-7.5 range)
# - original: STEAD dataset (original PhaseNet)
# - geofon: GEOFON data (global events)
# - scedc: Southern California Earthquake Data Center
PHASENET_MODEL = "stead"

DATA_DIR       = '/Users/francescoacolosimo/Desktop/SED/envelopes_test/data/maren_eq'
PROCESSED_DIR  = '/Users/francescoacolosimo/Desktop/SED/envelopes_test/data/operational_processed'  # output of 01_process_obs_mseed.py  → input here
ENVELOPES_DIR  = '/Users/francescoacolosimo/Desktop/SED/envelopes_test/data/operational_envelopes'   # output of this script              → input to 03

# Unified template library (used by 03_combined_CUA_synthetic_heatmap_comparison.py)
# Structure: SYN_CUA_ENV / R|S / mag / dist / {CUA_H.npy, CUA_Z.npy, median_envelope_*.npy}
SYN_CUA_ENV    = '/Users/francescoacolosimo/Desktop/SED/envelopes_test/data/syn_cua_env'
# ============================================================================

"""
Compute vertical and combined horizontal envelopes per station from a processed
miniSEED file and save them in a directory structure:

  <output_dir>/<NETWORK.STATION>/Vertical/envelope.npy
  <output_dir>/<NETWORK.STATION>/Horizontal_combined/envelope.npy

This follows the envelope logic present in `envelope_source.txt` and the
existing helper functions.

Usage:
  python create_envelopes_chino_hills.py --mseed /path/to/chino_hills_processed.ms \
      --output /path/to/output_dir --limit-stations 1

The script is conservative about assumptions:
- Station identifier used is NETWORK.STATION (e.g. CI.ASCO)
- Horizontal components are identified by channel not ending with 'Z'
- Vertical is identified by channel ending with 'Z'
- Envelopes are computed as maximum absolute value per second (integer seconds)

"""
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy
import scipy.signal as _scipy_signal
import obspy
from obspy import read, Stream
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import seisbench.models as sbm
import matplotlib.pyplot as plt


def envelope_of_trace(trace):
    """Compute envelope as max(abs(data)) per-second for a given obspy.Trace.
    Returns tuple: (envelope_array, absolute_starttime)
    """
    data = np.abs(trace.data)
    sr = int(round(trace.stats.sampling_rate))
    if sr <= 0:
        raise ValueError(f"Invalid sampling rate {trace.stats.sampling_rate} for {trace.id}")
    n_full_secs = len(data) // sr
    if n_full_secs == 0:
        # If trace shorter than 1s, just return max value
        return np.array([np.max(data)]), trace.stats.starttime
    env = [np.max(data[i * sr:(i + 1) * sr]) for i in range(n_full_secs)]
    return np.array(env), trace.stats.starttime


def calc_root_mean_squared_comb(h1, h2):
    """RMS combine two numpy arrays; truncates to shorter length."""
    comb_len = min(len(h1), len(h2))
    if comb_len == 0:
        return np.array([])
    h1s = np.array(h1[:comb_len], dtype=float)
    h2s = np.array(h2[:comb_len], dtype=float)
    combined = np.sqrt((h1s**2 + h2s**2) / 2.0)
    return combined


def compute_p_travel_time_seconds(distance_km, depth_km, model):
    """Return P-wave travel time in seconds using TauPyModel (iasp91).
    For very short distances (<1 degree / ~111 km), TauP may not work well,
    so we use a simple velocity model: hypocentral_dist / 6.0 km/s as fallback.
    distance_km -> degrees (approx 111.19 km per degree)."""
    try:
        deg = float(distance_km) / 111.19
        # For very short distances, TauP may not return arrivals - use simple velocity model
        if deg < 1.0:
            # Simple crustal P-wave velocity: 6.0 km/s
            # Hypocentral distance = sqrt(epicentral_distance^2 + depth^2)
            # But we're already given hypocentral_distance_km
            p_velocity_km_s = 6.0
            travel_time = float(distance_km) / p_velocity_km_s
            return travel_time
        
        arrivals = model.get_travel_times(source_depth_in_km=float(depth_km),
                                          distance_in_degree=deg,
                                          phase_list=["P"])
        if arrivals:
            return arrivals[0].time
        # If no arrivals even for longer distance, fallback to velocity model
        p_velocity_km_s = 6.0
        travel_time = float(distance_km) / p_velocity_km_s
        return travel_time
    except Exception:
        # Last resort fallback
        try:
            p_velocity_km_s = 6.0
            return float(distance_km) / p_velocity_km_s
        except:
            return None


def crop_envelope_by_p(env_array, trace_starttime, event_origin_time, p_travel_time, window_length_s=60):
    """Store P-wave timing metadata for envelope WITHOUT cropping to maintain absolute time.
    trace_starttime and event_origin_time are UTCDateTime objects. env_array is per-second.
    Returns original envelope array and metadata (dict) with absolute timing information.
    """
    if p_travel_time is None:
        return env_array, {
            "error": "no_p_travel_time",
            "envelope_starttime": str(trace_starttime),
            "event_origin_time": str(event_origin_time),
            "envelope_length_s": int(len(env_array))
        }

    p_arrival_time = event_origin_time + float(p_travel_time)
    offset_s = (p_arrival_time - trace_starttime)
    # Calculate where P-wave would be in envelope indices
    start_idx = int(np.floor(offset_s))
    if start_idx < 0:
        start_idx = 0
    end_idx = start_idx + int(window_length_s)
    if start_idx >= len(env_array):
        return env_array, {
            "error": "p_after_end_of_envelope", 
            "envelope_starttime": str(trace_starttime),
            "event_origin_time": str(event_origin_time),
            "p_travel_time_s": float(p_travel_time),
            "p_arrival_time": str(p_arrival_time),
            "start_idx": start_idx, 
            "envelope_length_s": int(len(env_array))
        }
    # Return FULL envelope with metadata about P-wave location
    meta = {
        "envelope_starttime": str(trace_starttime),
        "event_origin_time": str(event_origin_time),
        "p_travel_time_s": float(p_travel_time),
        "p_arrival_time": str(p_arrival_time),
        "p_arrival_idx_in_envelope": int(start_idx),
        "suggested_window_end_idx": int(min(end_idx, len(env_array))),
        "suggested_window_s": int(window_length_s),
        "envelope_length_s": int(len(env_array))
    }
    return env_array, meta


def pick_p_with_phasenet(stream, station_code, event_origin_time, model, taup_p_tt=None, distance_km=None):
    """PhaseNet picker using 3-component waveform data, constrained by TauP prediction.
    
    Parameters:
    - stream: ObsPy Stream containing all traces
    - station_code: Station code in format NETWORK.STATION
    - event_origin_time: UTCDateTime of event origin
    - model: Loaded PhaseNet model from seisbench
    - taup_p_tt: TauP theoretical P-wave travel time (optional, for constraining picks)
    - distance_km: Hypocentral distance in km (optional, for validation)
    
    Returns:
    - tuple: (pick_time, confidence_data) or (None, None) on failure
      where confidence_data is a dict with P/S/N probabilities and metadata
    """
    try:
        network, station = station_code.split('.')
        tr_z = stream.select(network=network, station=station, channel='*Z')
        tr_n = stream.select(network=network, station=station, channel='*N')
        tr_e = stream.select(network=network, station=station, channel='*E')
        
        if len(tr_z) == 0 or len(tr_n) == 0 or len(tr_e) == 0:
            return None, None
        
        # Create 3-component stream with preprocessing
        st_3c = Stream()
        for tr in [tr_z[0], tr_n[0], tr_e[0]]:
            tr_copy = tr.copy()
            tr_copy.detrend(type='demean')
            tr_copy.taper(max_percentage=0.05)
            
            # Resample to 100 Hz (PhaseNet requirement)
            target_sr = 100.0
            if tr_copy.stats.sampling_rate != target_sr:
                tr_copy.resample(target_sr)
            
            # Apply bandpass filter to reduce noise
            # P-waves typically 1-15 Hz for regional events
            try:
                tr_copy.filter('bandpass', freqmin=1.0, freqmax=15.0, corners=4, zerophase=True)
            except Exception:
                # Skip filtering if it fails
                pass
            
            st_3c.append(tr_copy)
        
        # Use classify method with threshold
        picks = model.classify(st_3c, P_threshold=0.3, S_threshold=0.5)
        p_picks = [p for p in picks.picks if p.phase == 'P']
        
        if len(p_picks) == 0:
            return None, {'error': 'no_picks_above_threshold'}
        
        # If TauP prediction available, filter picks to those near TauP
        best_pick = None
        if taup_p_tt is not None:
            # Define acceptable window: TauP ± 10 seconds
            taup_window = 10.0
            valid_picks = []
            for pick in p_picks:
                pick_tt = float(pick.peak_time - event_origin_time)
                diff_from_taup = abs(pick_tt - taup_p_tt)
                if diff_from_taup <= taup_window:
                    valid_picks.append((pick, diff_from_taup))
            
            if len(valid_picks) > 0:
                # Select pick closest to TauP with highest confidence
                valid_picks.sort(key=lambda x: (x[1], -x[0].peak_value))  # Sort by distance from TauP, then by confidence
                best_pick = valid_picks[0][0]
                print(f"    PhaseNet: Selected pick {float(best_pick.peak_time - event_origin_time):.1f}s (TauP: {taup_p_tt:.1f}s, diff: {valid_picks[0][1]:.1f}s)")
            else:
                # No picks within TauP window, reject all
                confidence_data = {
                    'error': 'no_picks_near_taup',
                    'taup_p_tt': taup_p_tt,
                    'n_picks_found': len(p_picks),
                    'closest_pick_diff': min([abs(float(p.peak_time - event_origin_time) - taup_p_tt) for p in p_picks])
                }
                print(f"    PhaseNet: All {len(p_picks)} picks outside TauP window (±{taup_window}s)")
                return None, confidence_data
        else:
            # No TauP constraint, use first (highest confidence) pick
            best_pick = p_picks[0]
        
        pick_time = best_pick.peak_time
        travel_time = pick_time - event_origin_time
        
        # Get confidence from pick probability
        p_confidence = float(best_pick.peak_value)
        
        # Store confidence data for analysis
        confidence_data = {
            'p_max_confidence': p_confidence,
            'pick_time': str(pick_time),
            'travel_time_s': float(travel_time),
            'trace_starttime': str(st_3c[0].stats.starttime),
            'n_picks_found': len(p_picks),
            'taup_p_tt': taup_p_tt,
            'taup_constrained': taup_p_tt is not None
        }
        
        # Validate pick is physically reasonable
        if travel_time < 0:
            confidence_data['rejected'] = True
            confidence_data['rejection_reason'] = f'travel_time={travel_time:.1f}s < 0 (pick before origin)'
            return None, confidence_data
        
        if travel_time > 300:
            confidence_data['rejected'] = True
            confidence_data['rejection_reason'] = f'travel_time={travel_time:.1f}s > 300s'
            return None, confidence_data
        
        # Additional validation with TauP if available
        if taup_p_tt is not None:
            diff_from_taup = abs(travel_time - taup_p_tt)
            confidence_data['taup_diff_s'] = diff_from_taup
        
        confidence_data['rejected'] = False
        return pick_time, confidence_data
        
    except Exception as e:
        return None, {'error': str(e)}


def pick_p_with_aic(trace, event_origin_time=None, distance_km=None, depth_km=None, taup_tt=None):
    """Pick P arrival using Akaike Information Criterion (AIC) method.
    Searches within a physically plausible window based on TauP or simple velocity model.
    
    Parameters:
    - trace: ObsPy trace
    - event_origin_time: UTCDateTime of event origin (optional, for constraining search)
    - distance_km: Hypocentral distance in km (optional, for velocity-based window)
    - depth_km: Source depth in km (optional, for velocity-based window)
    - taup_tt: TauP theoretical travel time (optional, for tighter window)
    
    Returns:
    - UTCDateTime of pick or None on failure
    """
    try:
        tr = trace.copy()
        tr.detrend(type='demean')
        tr.taper(max_percentage=0.05)
        
        fs = tr.stats.sampling_rate
        # Apply P-wave optimized bandpass filter
        if fs >= 40:
            try:
                tr.filter('bandpass', freqmin=2.0, freqmax=15.0, corners=4, zerophase=True)
            except Exception:
                pass
        
        data = np.array(tr.data, dtype=float)
        if data.size < int(5 * fs):  # Need at least 5 seconds
            return None
        
        # Define search window based on TauP or velocity model
        if taup_tt is not None:
            # Use tight window around TauP prediction
            min_travel_time = taup_tt - 7.0
            max_travel_time = taup_tt + 7.0
        elif event_origin_time is not None and distance_km is not None:
            # Simple velocity model for P-waves
            # Crustal P-wave velocity: ~6 km/s (range 5.5-7 km/s)
            # Upper mantle P-wave velocity: ~8 km/s
            # Use conservative bounds: 5.0 km/s (slow) to 8.5 km/s (fast)
            v_slow = 5.0  # km/s - slowest reasonable P-wave
            v_fast = 8.5  # km/s - fastest reasonable P-wave
            
            # Calculate expected travel time range
            min_travel_time = distance_km / v_fast  # Earliest possible arrival
            max_travel_time = distance_km / v_slow  # Latest reasonable arrival
            
            # Add safety margins: -3s before, +5s after expected window
            min_travel_time = max(0, min_travel_time - 3.0)
            max_travel_time = max_travel_time + 5.0
            
            # Convert to absolute times
            search_start_time = event_origin_time + min_travel_time
            search_end_time = event_origin_time + max_travel_time
            
            # Ensure search window is within trace bounds
            search_start_time = max(tr.stats.starttime + 0.5, search_start_time)
            search_end_time = min(tr.stats.endtime - 0.5, search_end_time)
            
            start_idx = int((search_start_time - tr.stats.starttime) * fs)
            end_idx = int((search_end_time - tr.stats.starttime) * fs)
        else:
            # Fallback: Search in middle portion of trace
            start_idx = int(5.0 * fs)
            end_idx = min(len(data) - int(5.0 * fs), start_idx + int(120 * fs))
        
        # Ensure valid indices
        start_idx = max(int(1.0 * fs), start_idx)
        end_idx = min(len(data) - int(1.0 * fs), end_idx)
        
        if end_idx - start_idx < int(2 * fs):  # Need at least 2s window
            return None
        
        # Compute AIC function using characteristic function (squared data)
        cf = data ** 2
        
        # Decimate to reduce computation: use every Nth sample
        # For 100 Hz, use every 5th sample -> 20 Hz effective
        decimate_factor = max(1, int(fs / 20))
        search_range = range(start_idx, end_idx, decimate_factor)
        
        aic_values = []
        indices = []
        
        for k in search_range:
            if k < 100 or k > len(cf) - 100:  # Need buffer
                continue
            # Compute variance of windows before and after k
            var1 = np.var(cf[:k])
            var2 = np.var(cf[k:])
            if var1 > 1e-20 and var2 > 1e-20:  # Avoid log(0)
                aic_val = k * np.log(var1) + (len(cf) - k - 1) * np.log(var2)
                aic_values.append(aic_val)
                indices.append(k)
        
        if len(aic_values) == 0:
            return None
        
        # Find minimum AIC
        min_idx = indices[np.argmin(aic_values)]
        pick_time = tr.stats.starttime + float(min_idx) / fs
        
        # Final validation if event origin time is provided
        if event_origin_time is not None:
            travel_time = pick_time - event_origin_time
            # Reject if pick is before origin or unreasonably late (>200s)
            if travel_time < 0 or travel_time > 200:
                return None
        
        return pick_time
    except Exception:
        return None


def process_station(traces, out_base, station_df=None, model=None, event_origin_time=None, event_depth_km=None, 
                   full_stream=None, phasenet_model=None):
    """Given a list of obspy.Trace objects belonging to the same station,
    compute vertical envelope(s) and combined horizontal envelope and save them.
    Returns a dict with pick information for CSV export.
    
    Parameters:
    - traces: List of traces for this station
    - out_base: Output base directory
    - station_df: DataFrame with station distances
    - model: TauP model for theoretical arrivals
    - event_origin_time: Event origin time
    - event_depth_km: Event depth in km
    - full_stream: Full ObsPy Stream (needed for PhaseNet 3-component picking)
    - phasenet_model: Loaded PhaseNet model from seisbench
    """
    # station id
    if not traces:
        return None
    network = traces[0].stats.network
    station = traces[0].stats.station
    station_code = f"{network}.{station}"

    station_dir = Path(out_base) / station_code
    vert_dir = station_dir / "Vertical"
    horiz_dir = station_dir / "Horizontal_combined"
    vert_dir.mkdir(parents=True, exist_ok=True)
    horiz_dir.mkdir(parents=True, exist_ok=True)

    # storage for station-level pick info (to reuse for horizontal)
    station_pick = {
        'p_travel_time_used': None,
        'pick_method': None,
        'phasenet_pick_time': None,
        'aic_pick_time': None,
        'taup_p_tt': None,
        'phasenet_taup_diff_s': None,
        'aic_taup_diff_s': None
    }

    # Vertical traces: channel ending with 'Z'
    vertical_traces = [tr for tr in traces if tr.stats.channel[-1] == 'Z']
    if vertical_traces:
        # If multiple vertical traces, compute envelope for each and save separately
        for i, vt in enumerate(vertical_traces):
            env, env_starttime = envelope_of_trace(vt)
            # If station_df and model provided, compute P arrival and add timing metadata
            meta = {
                "envelope_starttime": str(env_starttime),
                "sampling_rate_hz": 1.0,  # Envelope is 1 sample per second
                "envelope_length_s": int(len(env))
            }
            if station_df is not None and model is not None and event_origin_time is not None and event_depth_km is not None:
                station_code = f"{vt.stats.network}.{vt.stats.station}"
                row = station_df[station_df['station_code'] == station_code]
                if not row.empty:
                    dist_km = float(row.iloc[0]['hypocentral_distance_km'])
                    taup_p_tt = compute_p_travel_time_seconds(dist_km, event_depth_km, model)
                    station_pick['taup_p_tt'] = taup_p_tt

                    # Try PhaseNet first (primary picker) - constrained by TauP
                    phasenet_time = None
                    phasenet_p_tt = None
                    phasenet_confidence = None
                    if phasenet_model is not None and full_stream is not None:
                        phasenet_time, phasenet_confidence = pick_p_with_phasenet(full_stream, station_code, event_origin_time, phasenet_model, taup_p_tt=taup_p_tt, distance_km=dist_km)
                        if phasenet_time is not None:
                            phasenet_p_tt = float(phasenet_time - event_origin_time)
                            station_pick['phasenet_pick_time'] = str(phasenet_time)
                            station_pick['phasenet_confidence'] = phasenet_confidence['p_max_confidence'] if phasenet_confidence else None
                        # Save confidence data even if pick was rejected
                        if phasenet_confidence is not None:
                            confidence_path = vert_dir / f"phasenet_confidence_{vt.stats.channel}.json"
                            with open(confidence_path, 'w') as f:
                                json.dump(phasenet_confidence, f, indent=2)
                    
                    # Try AIC with TauP-guided window (fallback)
                    aic_time = pick_p_with_aic(vt, event_origin_time, distance_km=dist_km, depth_km=event_depth_km, taup_tt=taup_p_tt)
                    aic_p_tt = None
                    if aic_time is not None:
                        aic_p_tt = float(aic_time - event_origin_time)
                        station_pick['aic_pick_time'] = str(aic_time)

                    # Picking hierarchy: 
                    # 1. PhaseNet (if available and successful)
                    # 2. AIC close to TauP (if PhaseNet fails)
                    # 3. TauP theoretical only (if both fail)
                    pick_used = None
                    p_tt_used = None
                    
                    if phasenet_p_tt is not None:
                        # Use PhaseNet as primary picker
                        pick_used = 'phasenet'
                        p_tt_used = phasenet_p_tt
                        # Record comparison with TauP
                        if taup_p_tt is not None:
                            station_pick['phasenet_taup_diff_s'] = abs(phasenet_p_tt - taup_p_tt)
                            if aic_p_tt is not None:
                                station_pick['aic_taup_diff_s'] = abs(aic_p_tt - taup_p_tt)
                    elif aic_p_tt is not None:
                        # Use AIC if PhaseNet fails
                        pick_used = 'aic'
                        p_tt_used = aic_p_tt
                        if taup_p_tt is not None:
                            station_pick['aic_taup_diff_s'] = abs(aic_p_tt - taup_p_tt)
                    elif taup_p_tt is not None:
                        # Use TauP theoretical as last resort
                        pick_used = 'taup'
                        p_tt_used = taup_p_tt
                    else:
                        # No pick available
                        pick_used = None
                        p_tt_used = None

                    station_pick['p_travel_time_used'] = p_tt_used
                    station_pick['pick_method'] = pick_used

                    # Store timing metadata WITHOUT cropping (preserve absolute time)
                    env_with_timing, timing_meta = crop_envelope_by_p(env, env_starttime, event_origin_time, p_tt_used, window_length_s=60)
                    # add pick metadata
                    meta.update(timing_meta)
                    meta.update({
                        'pick_method': pick_used,
                        'phasenet_pick_time': station_pick.get('phasenet_pick_time'),
                        'aic_pick_time': station_pick.get('aic_pick_time'),
                        'taup_p_tt': station_pick.get('taup_p_tt'),
                        'phasenet_taup_diff_s': station_pick.get('phasenet_taup_diff_s'),
                        'aic_taup_diff_s': station_pick.get('aic_taup_diff_s')
                    })
                    env_to_save = env_with_timing
                else:
                    env_to_save = env
            else:
                env_to_save = env

            # filename with channel
            fname = vert_dir / f"envelope_{vt.stats.channel}.npy"
            np.save(fname, env_to_save)
            # Save metadata if present
            if meta:
                (vert_dir / f"envelope_{vt.stats.channel}_meta.json").write_text(json.dumps(meta, indent=2))
    else:
        # No vertical - log by creating placeholder
        (vert_dir / "MISSING_vertical.txt").write_text("No vertical (Z) traces found for this station")

    # Horizontal: traces not ending with Z. We expect two components (E/N or H? letter end)
    horiz_traces = [tr for tr in traces if tr.stats.channel[-1] != 'Z']
    if len(horiz_traces) >= 2:
        # Pick first two horizontal components (sorted by channel for determinism)
        horiz_traces = sorted(horiz_traces, key=lambda t: t.stats.channel)
        t1, t2 = horiz_traces[0], horiz_traces[1]
        # Preprocess like envelope_source: detrend/taper/highpass
        t1p = t1.copy(); t2p = t2.copy()
        try:
            t1p.detrend(type='demean'); t2p.detrend(type='demean')
            t1p.taper(0.05); t2p.taper(0.05)
            # Use a moderate highpass like 0.33 Hz as in source
            t1p.filter('highpass', freq=0.33)
            t2p.filter('highpass', freq=0.33)
        except Exception:
            # If filtering fails or not available, continue with raw
            pass
        # Combine then compute envelope per-second
        combined_ts = calc_root_mean_squared_comb(t1p.data, t2p.data)
        if combined_ts.size == 0:
            (horiz_dir / "MISSING_horizontal.txt").write_text("Horizontal combination empty after processing")
        else:
            # Now compute envelope per second for combined_ts: need sampling rate
            sr = int(round(t1p.stats.sampling_rate))
            if sr <= 0:
                (horiz_dir / "MISSING_horizontal.txt").write_text("Invalid sampling rate for horizontal traces")
            else:
                n_full_secs = len(combined_ts) // sr
                if n_full_secs == 0:
                    env = np.array([np.max(np.abs(combined_ts))])
                else:
                    env = np.array([np.max(np.abs(combined_ts[i*sr:(i+1)*sr])) for i in range(n_full_secs)])
                # Store timing metadata; reuse station vertical pick if available
                env_starttime = t1p.stats.starttime
                meta = {
                    "envelope_starttime": str(env_starttime),
                    "sampling_rate_hz": 1.0,  # Envelope is 1 sample per second
                    "envelope_length_s": int(len(env))
                }
                station_code = f"{t1p.stats.network}.{t1p.stats.station}"
                env_to_save = env
                if station_df is not None and model is not None and event_origin_time is not None and event_depth_km is not None:
                    row = station_df[station_df['station_code'] == station_code]
                    if not row.empty:
                        dist_km = float(row.iloc[0]['hypocentral_distance_km'])
                        # Prefer station-level pick computed from vertical
                        p_tt_used = station_pick.get('p_travel_time_used')
                        pick_method = station_pick.get('pick_method')
                        # If no station pick, attempt to pick on one of the horizontal traces
                        if p_tt_used is None:
                            taup_p_tt = compute_p_travel_time_seconds(dist_km, event_depth_km, model)
                            
                            # Try PhaseNet first
                            if phasenet_model is not None and full_stream is not None:
                                phasenet_time, phasenet_confidence = pick_p_with_phasenet(full_stream, station_code, event_origin_time, phasenet_model, taup_p_tt=taup_p_tt, distance_km=dist_km)
                                if phasenet_time is not None:
                                    phasenet_p_tt = float(phasenet_time - event_origin_time)
                                    pick_method = 'phasenet'
                                    p_tt_used = phasenet_p_tt
                                    if phasenet_confidence:
                                        meta['phasenet_confidence'] = phasenet_confidence['p_max_confidence']
                            
                            # Fallback to AIC with TauP window
                            if p_tt_used is None:
                                aic_time = pick_p_with_aic(t1, event_origin_time, distance_km=dist_km, depth_km=event_depth_km, taup_tt=taup_p_tt)
                                if aic_time is not None:
                                    aic_p_tt = float(aic_time - event_origin_time)
                                    pick_method = 'aic'
                                    p_tt_used = aic_p_tt
                            
                            # Last resort: use TauP theoretical
                            if p_tt_used is None and taup_p_tt is not None:
                                pick_method = 'taup'
                                p_tt_used = taup_p_tt
                            
                            if p_tt_used is None:
                                pick_method = None
                            
                            meta.update({
                                'pick_method': pick_method, 
                                'aic_pick_time': str(aic_time) if 'aic_time' in locals() and aic_time is not None else None,
                                'taup_p_tt': taup_p_tt
                            })
                        # Store timing metadata WITHOUT cropping (preserve absolute time)
                        if p_tt_used is not None:
                            env_with_timing, timing_meta = crop_envelope_by_p(env, env_starttime, event_origin_time, p_tt_used, window_length_s=60)
                            env_to_save = env_with_timing
                            meta.update(timing_meta)
                np.save(horiz_dir / f"envelope_{t1p.stats.channel}_{t2p.stats.channel}.npy", env_to_save)
                if meta:
                    (horiz_dir / f"envelope_{t1p.stats.channel}_{t2p.stats.channel}_meta.json").write_text(json.dumps(meta, indent=2))
    else:
        (horiz_dir / "MISSING_horizontal.txt").write_text("Less than 2 horizontal components available for combination")

    # Return pick information for CSV export
    pick_record = {
        'station_code': station_code,
        'network': network,
        'station': station,
        'pick_method': station_pick.get('pick_method'),
        'p_travel_time_used': station_pick.get('p_travel_time_used'),
        'phasenet_pick_time': station_pick.get('phasenet_pick_time'),
        'phasenet_confidence': station_pick.get('phasenet_confidence'),
        'aic_pick_time': station_pick.get('aic_pick_time'),
        'taup_p_tt': station_pick.get('taup_p_tt'),
        'phasenet_taup_diff_s': station_pick.get('phasenet_taup_diff_s'),
        'aic_taup_diff_s': station_pick.get('aic_taup_diff_s')
    }
    return pick_record


def main():
    # Extract short name from event_id and convert CamelCase to snake_case
    # e.g., "20140824_M6.0_SouthNapa" -> "south_napa", "20080729_M5.4_ChinoHills" -> "chino_hills"
    event_name = event_id.split('_')[-1]  # Get the last part (e.g., "SouthNapa", "ChinoHills")
    # Convert CamelCase to snake_case
    import re
    short_name = re.sub(r'(?<!^)(?=[A-Z])', '_', event_name).lower()
    
    # Construct file paths based on event ID
    default_mseed = os.path.join(PROCESSED_DIR, f'processed_traces_{short_name}', 'processed.ms')
    default_output = os.path.join(ENVELOPES_DIR, f'{short_name}_envelope')
    default_station_csv = os.path.join(DATA_DIR, f'station_distance_table_{event_id}.csv')
    default_event_json = os.path.join(DATA_DIR, f'{event_id}.json')
    
    parser = argparse.ArgumentParser(description="Create envelopes for processed traces")
    parser.add_argument('--mseed', required=False,
                        default=default_mseed,
                        help='Input processed miniSEED file')
    parser.add_argument('--output', required=False,
                        default=default_output,
                        help='Output base directory for station envelopes')
    parser.add_argument('--limit-stations', type=int, default=None,
                        help='Process only the first N stations (for quick tests)')
    parser.add_argument('--station-csv', required=False,
                        default=default_station_csv,
                        help='CSV with station distances and event info')
    parser.add_argument('--event-json', required=False,
                        default=default_event_json,
                        help='Event JSON with origin time and depth')

    args = parser.parse_args()
    
    print("=" * 80)
    print("ENVELOPE CREATION")
    print("=" * 80)
    print(f"Event ID: {event_id}")
    print(f"Short name: {short_name}")
    print()
    print(f"File paths:")
    print(f"  MSEED:       {args.mseed}")
    print(f"  Output:      {args.output}")
    print(f"  Station CSV: {args.station_csv}")
    print(f"  Event JSON:  {args.event_json}")
    print()
    
    # Check if files exist
    if not os.path.exists(args.mseed):
        print(f"❌ MSEED file not found: {args.mseed}")
        return
    if not os.path.exists(args.station_csv):
        print(f"⚠️  Station CSV not found: {args.station_csv}")
    if not os.path.exists(args.event_json):
        print(f"⚠️  Event JSON not found: {args.event_json}")
    print()

    print(f"Reading mseed: {args.mseed}")
    st = read(args.mseed)
    print(f"Total traces read: {len(st)}")

    # Group traces by station identifier NETWORK.STATION
    station_map = {}
    for tr in st:
        key = f"{tr.stats.network}.{tr.stats.station}"
        station_map.setdefault(key, []).append(tr)

    station_keys = sorted(station_map.keys())
    if args.limit_stations is not None:
        station_keys = station_keys[:args.limit_stations]

    # Load station CSV and event JSON if provided
    station_df = None
    event_origin_time = None
    event_depth_km = None
    model = None
    if args.station_csv and os.path.exists(args.station_csv):
        station_df = pd.read_csv(args.station_csv)
        print(f"Loaded station distance table: {args.station_csv} ({len(station_df)} rows)")
    else:
        print("No station CSV provided or file not found; envelopes will not be cropped by P arrival.")

    if args.event_json and os.path.exists(args.event_json):
        with open(args.event_json, 'r') as fh:
            try:
                ev = json.load(fh)
                # expect format with events[0].time and events[0].dep
                if 'events' in ev and len(ev['events']) > 0:
                    e0 = ev['events'][0]
                    event_origin_time = UTCDateTime(e0['time'])
                    event_depth_km = float(e0.get('dep', e0.get('depth', 15.5)))
                    print(f"Loaded event: origin={event_origin_time}, depth_km={event_depth_km}")
            except Exception as e:
                print(f"Failed to read event JSON: {e}")
    else:
        print("No event JSON provided or file not found; envelopes will not be cropped by P arrival.")

    if station_df is not None and event_origin_time is not None:
        try:
            model = TauPyModel(model='iasp91')
            print("Initialized TauPyModel iasp91")
        except Exception as e:
            print(f"Failed to initialize TauPyModel: {e}")

    # Load PhaseNet model (instance variant trained on M5-7.5 range)
    phasenet_model = None
    try:
        print(f"\nLoading PhaseNet model ({PHASENET_MODEL})...")
        phasenet_model = sbm.PhaseNet.from_pretrained(PHASENET_MODEL)
        print(f"✓ PhaseNet model '{PHASENET_MODEL}' loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load PhaseNet model '{PHASENET_MODEL}': {e}")
        print("   Will use AIC/STA-LTA fallback only")

    # Store pick information for all stations
    pick_records = []

    print(f"\nProcessing {len(station_keys)} stations")
    for sk in station_keys:
        pick_info = process_station(station_map[sk], args.output, station_df=station_df, model=model, 
                                    event_origin_time=event_origin_time, event_depth_km=event_depth_km,
                                    full_stream=st, phasenet_model=phasenet_model)
        if pick_info:
            pick_records.append(pick_info)
        print(f"Processed station {sk}")

    # Save pick information to CSV
    if pick_records:
        picks_csv_path = Path(args.output) / "station_picks.csv"
        picks_df = pd.DataFrame(pick_records)
        picks_df.to_csv(picks_csv_path, index=False)
        print(f"\nSaved pick information to: {picks_csv_path}")
        
        # Print statistics
        print("\n" + "="*80)
        print(f"PICKING STATISTICS - PhaseNet Model: {PHASENET_MODEL}")
        print("="*80)
        total_stations = len(picks_df)
        
        method_counts = picks_df['pick_method'].value_counts()
        print(f"\nTotal stations processed: {total_stations}")
        print(f"\nPicks by method:")
        for method, count in method_counts.items():
            percentage = (count / total_stations) * 100
            print(f"  {method:12s}: {count:4d} stations ({percentage:5.1f}%)")
        
        # Count stations with no pick
        no_pick = picks_df['pick_method'].isna().sum()
        if no_pick > 0:
            percentage = (no_pick / total_stations) * 100
            print(f"  {'No pick':12s}: {no_pick:4d} stations ({percentage:5.1f}%)")
        
        # Success rate
        success_count = total_stations - no_pick
        success_rate = (success_count / total_stations) * 100
        print(f"\nOverall picking success rate: {success_count}/{total_stations} ({success_rate:.1f}%)")
        print("="*80)

        # Plot sample of waveforms with picks for visual QC
        print("\nGenerating quick visual QC plots...")
        plot_pick_samples(st, picks_df, args.output, event_origin_time, n_samples=20)

    print("\nDone. Envelopes saved under:", args.output)


def plot_pick_samples(stream, picks_df, envelope_dir, origin_time, n_samples=20):
    """Plot a sample of waveforms with picks for quick visual QC."""
    import matplotlib.pyplot as plt
    import random
    
    # Select stations: prioritize those with picks
    picked_stations = picks_df[picks_df['pick_method'].notna()]['station_code'].tolist()
    unpicked_stations = picks_df[picks_df['pick_method'].isna()]['station_code'].tolist()
    
    n_picked = min(n_samples // 2, len(picked_stations))
    n_unpicked = min(n_samples - n_picked, len(unpicked_stations))
    
    stations_to_plot = (
        random.sample(picked_stations, n_picked) if picked_stations else []
    ) + (
        random.sample(unpicked_stations, n_unpicked) if unpicked_stations else []
    )
    
    if not stations_to_plot:
        print("  No stations available to plot")
        return
    
    # Create subplot grid
    n_cols = 4
    n_rows = int(np.ceil(len(stations_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), sharex=True)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, station_code in enumerate(stations_to_plot):
        ax = axes[idx]
        station_row = picks_df[picks_df['station_code'] == station_code].iloc[0]
        network, station = station_code.split('.')
        
        # Get vertical trace
        tr_z = stream.select(network=network, station=station, channel='*Z')
        if len(tr_z) == 0:
            ax.text(0.5, 0.5, f'{station_code}\nNo Z trace', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        tr = tr_z[0].copy()
        tr.detrend('demean')
        
        # Plot waveform
        times = tr.times(type='relative', reftime=origin_time)
        ax.plot(times, tr.data, 'k', linewidth=0.3, alpha=0.7)
        
        # Plot origin time
        ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Plot picks
        if pd.notna(station_row['phasenet_pick_time']):
            phasenet_time = UTCDateTime(station_row['phasenet_pick_time'])
            phasenet_rel = phasenet_time - origin_time
            ax.axvline(phasenet_rel, color='blue', linewidth=1.2, alpha=0.7)
        
        if pd.notna(station_row['aic_pick_time']):
            aic_time = UTCDateTime(station_row['aic_pick_time'])
            aic_rel = aic_time - origin_time
            ax.axvline(aic_rel, color='green', linewidth=1.2, alpha=0.7)
        
        if pd.notna(station_row['taup_p_tt']):
            ax.axvline(station_row['taup_p_tt'], color='orange', 
                      linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Title with pick info
        pick_method = station_row['pick_method'] if pd.notna(station_row['pick_method']) else 'none'
        title = f"{station_code}\n{pick_method}"
        if pd.notna(station_row.get('phasenet_confidence')):
            title += f" (c={station_row['phasenet_confidence']:.2f})"
        ax.set_title(title, fontsize=8)
        
        ax.set_xlim(-5, 60)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    
    # Hide extra subplots
    for idx in range(len(stations_to_plot), len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Origin'),
        Line2D([0], [0], color='blue', label='PhaseNet'),
        Line2D([0], [0], color='green', label='AIC'),
        Line2D([0], [0], color='orange', linestyle='--', label='TauP')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
              bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(f'Pick Quality Check - {len(stations_to_plot)} Random Stations', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.show()
    print(f"  Plotted {len(stations_to_plot)} stations")


if __name__ == '__main__':
    main()
