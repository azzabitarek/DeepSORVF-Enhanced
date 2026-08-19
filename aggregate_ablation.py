"""
aggregate_ablation.py — Aggregate ablation results into metrics tables.

Reads ablation_log.csv files produced by run_ablation.py and computes:
  - Fusion rate (proportion of AIS MMSIs successfully locked)
  - ID switches (distinct visual IDs linked to each MMSI)
  - Lock latency (time from first sighting to first lock)

Also detects occlusion candidates from gap analysis.

Usage:
    python aggregate_ablation.py                          # aggregate latest run
    python aggregate_ablation.py --run-dir ./ablation_results/20260819_120000
"""
import os
import sys
import glob
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent


def load_expected_mmsis(ais_dir):
    """
    Extract all MMSI values present in a sequence's AIS data.
    Reads ALL per-second CSVs in the ais/ directory.
    """
    ais_path = Path(ais_dir)
    if not ais_path.exists():
        return set()

    mmsis = set()
    for csv_file in ais_path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file, usecols=[0], header=0)
            mmsis.update(df.iloc[:, 0].dropna().unique())
        except Exception:
            continue
    return mmsis


def compute_metrics(ablation_log_csv, ais_dir):
    """
    Compute fusion metrics from ablation log and AIS ground truth.

    Returns dict with:
      - fusion_rate: float [0..1]
      - id_switches: int
      - avg_latency_s: float or None
      - n_locked: int (number of MMSIs locked)
      - n_expected: int (number of MMSIs in AIS)
    """
    log = pd.read_csv(ablation_log_csv)
    expected_mmsis = load_expected_mmsis(ais_dir)

    if log.empty or not expected_mmsis:
        return {
            'fusion_rate': 0.0,
            'id_switches': 0,
            'avg_latency_s': None,
            'n_locked': 0,
            'n_expected': len(expected_mmsis),
        }

    # Fusion rate: proportion of expected MMSIs that got locked at least once
    locked_mmsis = set(log[log['is_new_lock'] == True]['mmsi'].unique())
    n_locked = len(locked_mmsis & expected_mmsis)
    n_expected = len(expected_mmsis)
    fusion_rate = n_locked / max(n_expected, 1)

    # ID switches: for each MMSI, count distinct visual IDs linked to it
    id_switches = 0
    for mmsi, group in log.groupby('mmsi'):
        distinct_ids = group['ID'].nunique()
        if distinct_ids > 1:
            id_switches += distinct_ids - 1

    # Lock latency: time from first sighting to first lock (in seconds)
    latencies = []
    for mmsi, group in log.groupby('mmsi'):
        first_seen = group['timestamp'].min()
        first_lock = group[group['is_new_lock'] == True]['timestamp'].min()
        if pd.notna(first_lock) and pd.notna(first_seen):
            latencies.append((first_lock - first_seen) / 1000.0)  # ms -> s
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        'fusion_rate': round(fusion_rate, 4),
        'id_switches': id_switches,
        'avg_latency_s': round(avg_latency, 2) if avg_latency is not None else None,
        'n_locked': n_locked,
        'n_expected': n_expected,
    }


def detect_occlusion_candidates(log_csv, gap_threshold_ms=2000):
    """
    Detect occlusion candidates from gaps in the ablation log.
    A gap > gap_threshold_ms between consecutive timestamps for the same
    MMSI suggests the vessel was occluded or lost.
    """
    if not os.path.exists(log_csv):
        return []

    log = pd.read_csv(log_csv)
    if log.empty:
        return []

    candidates = []
    for mmsi, group in log.groupby('mmsi'):
        timestamps = sorted(group['timestamp'].unique())
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > gap_threshold_ms:
                candidates.append({
                    'mmsi': int(mmsi),
                    'gap_start': int(timestamps[i]),
                    'gap_end': int(timestamps[i + 1]),
                    'gap_s': round(gap / 1000.0, 1),
                })

    return sorted(candidates, key=lambda c: -c['gap_s'])


def aggregate_run(run_dir, sequences, configs):
    """
    Aggregate results from a single ablation run directory.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory (e.g. ablation_results/20260819_120000/)
    sequences : list of str
        Sequence names to process
    configs : list of str
        Config IDs to process

    Returns
    -------
    per_sequence_df : DataFrame with per-sequence results
    summary_df : DataFrame with mean+std per config
    """
    rows = []
    occlusion_rows = []

    for seq in sequences:
        ais_dir = PROJECT_ROOT / seq / 'ais'
        expected_mmsis = load_expected_mmsis(ais_dir)

        for config in configs:
            config_dir = run_dir / seq / config

            # Find the ablation log file
            log_pattern = str(config_dir / f'*_ablation_log.csv')
            log_files = glob.glob(log_pattern)

            log_csv = log_files[0] if log_files else None

            # Read detection/tracking/fusion counts from metric files
            metric_dir = config_dir / 'metric'
            det_count = 0
            trk_count = 0
            fus_count = 0

            det_file = metric_dir / f'{seq}_detection.txt'
            trk_file = metric_dir / f'{seq}_tracking.txt'
            fus_file = metric_dir / f'{seq}_fusion.txt'

            if det_file.exists():
                with open(det_file) as f:
                    det_count = sum(1 for _ in f)
            if trk_file.exists():
                with open(trk_file) as f:
                    trk_count = sum(1 for _ in f)
            if fus_file.exists():
                with open(fus_file) as f:
                    fus_count = sum(1 for _ in f)

            if log_csv:
                metrics = compute_metrics(log_csv, ais_dir)
            else:
                metrics = {
                    'fusion_rate': 0.0,
                    'id_switches': 0,
                    'avg_latency_s': None,
                    'n_locked': 0,
                    'n_expected': len(expected_mmsis),
                }

            # Compute detection-level fusion rate: fus_count / det_count
            detection_fusion_rate = fus_count / max(det_count, 1)

            rows.append({
                'sequence': seq,
                'config': config,
                'detection_count': det_count,
                'tracking_count': trk_count,
                'fusion_count': fus_count,
                'detection_fusion_rate': round(detection_fusion_rate, 4),
                **metrics
            })

            # Occlusion candidates for fusion configs
            if config in ('C3', 'C4', 'C5', 'C3a', 'C3b', 'C3c', 'C3d'):
                candidates = detect_occlusion_candidates(log_csv)
                for c in candidates[:6]:
                    occlusion_rows.append({
                        'sequence': seq,
                        'config': config,
                        **c
                    })

    per_sequence_df = pd.DataFrame(rows)

    if not per_sequence_df.empty:
        agg_dict = {
            'detection_count': ('detection_count', 'mean'),
            'tracking_count': ('tracking_count', 'mean'),
            'fusion_count': ('fusion_count', 'mean'),
            'detection_fusion_rate': ('detection_fusion_rate', 'mean'),
            'fusion_rate_mean': ('fusion_rate', 'mean'),
            'fusion_rate_std': ('fusion_rate', 'std'),
            'id_switches_mean': ('id_switches', 'mean'),
            'id_switches_std': ('id_switches', 'std'),
            'avg_latency_s_mean': ('avg_latency_s', 'mean'),
            'avg_latency_s_std': ('avg_latency_s', 'std'),
        }
        summary_df = per_sequence_df.groupby('config').agg(**agg_dict).round(4)
    else:
        summary_df = pd.DataFrame()

    return per_sequence_df, summary_df, pd.DataFrame(occlusion_rows)


def main():
    parser = argparse.ArgumentParser(description='Aggregate ablation results')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Specific run directory (default: latest)')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Sequences to include')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Configs to include')
    args = parser.parse_args()

    result_root = PROJECT_ROOT / 'ablation_results'

    # Find the run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Use the latest run
        runs = sorted([d for d in result_root.iterdir() if d.is_dir()])
        if not runs:
            print("[ERROR] No ablation runs found in", result_root)
            sys.exit(1)
        run_dir = runs[-1]

    print(f"Aggregating results from: {run_dir}")

    # Discover sequences and configs from directory structure
    sequences = []
    for d in run_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            sequences.append(d.name)

    if args.sequences:
        sequences = [s for s in sequences if s in args.sequences]

    configs = set()
    for seq_dir in run_dir.iterdir():
        if seq_dir.is_dir():
            for cfg_dir in seq_dir.iterdir():
                if cfg_dir.is_dir():
                    configs.add(cfg_dir.name)
    configs = sorted(configs)
    if args.configs:
        configs = [c for c in configs if c in args.configs]

    print(f"  Sequences: {sequences}")
    print(f"  Configs:   {configs}")

    per_seq_df, summary_df, occlusion_df = aggregate_run(run_dir, sequences, configs)

    # Write CSVs
    per_seq_path = run_dir / 'ablation_per_sequence.csv'
    summary_path = run_dir / 'ablation_summary.csv'
    occlusion_path = run_dir / 'occlusion_candidates_to_validate.csv'

    per_seq_df.to_csv(per_seq_path, index=False)
    summary_df.to_csv(summary_path)
    if not occlusion_df.empty:
        occlusion_df.to_csv(occlusion_path, index=False)

    print(f"\n{'=' * 60}")
    print("  Ablation Summary (mean ± std across sequences)")
    print("=" * 60)
    if not summary_df.empty:
        print(summary_df.to_string())
    else:
        print("  No results found.")
    print(f"\n  Per-sequence: {per_seq_path}")
    print(f"  Summary:      {summary_path}")
    if not occlusion_df.empty:
        print(f"  Occlusions:   {occlusion_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
