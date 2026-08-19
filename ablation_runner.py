"""
ablation_runner.py — Batch runner for all ablation configurations.

Executes 6 primary configs (C0-C5) + 4 optional configs (C3a-C3d)
on all available test sequences. Results are written to ablation_results/.

Usage:
    python ablation_runner.py                        # run all
    python ablation_runner.py --configs C0 C1 C5     # run specific configs
    python ablation_runner.py --sequences clip-01     # run on specific sequences
    python ablation_runner.py --max-frames 50         # limit frames (for testing)
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_ablation import run_pipeline

# ── Configuration definitions ────────────────────────────────────────────────
# Each config is a dict of flags passed to run_pipeline().
# Default values (production behavior): all flags True, anti=1.
# C0-C5: Primary ablation configs (mandatory for the article table)
# C3a-C3d: Secondary configs (optional, for fine-grained discussion)

CONFIGS = {
    # ── Primary configs (mandatory) ──
    'C0': dict(use_ensemble=False, use_static_filter=False,
               ais_enabled=False, anti=0),                    # YOLOX only
    'C1': dict(use_ensemble=True,  use_static_filter=False,
               ais_enabled=False, anti=0),                    # + KOLOMVERSE ensemble
    'C2': dict(use_ensemble=True,  use_static_filter=True,
               ais_enabled=False, anti=0),                    # + static filter
    'C3': dict(use_ensemble=True,  use_static_filter=True,
               ais_enabled=True,  anti=0),                    # + AIS fusion (no OAR)
    'C4': dict(use_ensemble=True,  use_static_filter=True,
               ais_enabled=True,  anti=1),                    # + OAR
    'C5': dict(use_ensemble=True,  use_static_filter=True,
               ais_enabled=True,  anti=1),                    # Full model (= C4)

    # ── Secondary configs (optional) ──
    'C3a': dict(use_ensemble=True, use_static_filter=True,
                ais_enabled=True, anti=0, use_dtw=False),             # No DTW
    'C3b': dict(use_ensemble=True, use_static_filter=True,
                ais_enabled=True, anti=0, use_angle_penalty=False),   # No angular penalty
    'C3c': dict(use_ensemble=True, use_static_filter=True,
                ais_enabled=True, anti=0, use_binding=False),         # No binding
    'C3d': dict(use_ensemble=True, use_static_filter=True,
                ais_enabled=True, anti=0, use_hungarian=False),       # Greedy assignment
}

# ── Available sequences ──────────────────────────────────────────────────────
SEQUENCES = ['clip-01', 'clip-02', 'clip-10', 'Video-10', 'Video-28', 'Video-29', 'Video-34']

# ── Directories ──────────────────────────────────────────────────────────────
RESULT_ROOT = PROJECT_ROOT / 'ablation_results'


def main():
    parser = argparse.ArgumentParser(description='Ablation study batch runner')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Config IDs to run (default: all)')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Sequences to process (default: all)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit frames per sequence (for quick testing)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip configs that already have results')
    args = parser.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())
    sequences_to_run = args.sequences or SEQUENCES

    # Validate config names
    for c in configs_to_run:
        if c not in CONFIGS:
            print(f"[ERROR] Unknown config '{c}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)

    # Validate sequences exist
    for seq in sequences_to_run:
        seq_path = PROJECT_ROOT / 'data' / 'clips' / seq
        if not seq_path.exists():
            print(f"[WARNING] Sequence '{seq}' not found at {seq_path}, skipping")
            sequences_to_run.remove(seq)

    print("=" * 60)
    print("  DeepSORVF Ablation Study")
    print("=" * 60)
    print(f"  Configs:   {configs_to_run}")
    print(f"  Sequences: {sequences_to_run}")
    print(f"  Max frames: {args.max_frames or 'all'}")
    print(f"  Output:    {RESULT_ROOT}")
    print("=" * 60)

    # Create run directory with timestamp
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    total_start = time.time()

    for seq in sequences_to_run:
        print(f"\n{'─' * 40}")
        print(f"  Sequence: {seq}")
        print(f"{'─' * 40}")

        for config_name in configs_to_run:
            # Check if already done
            if args.skip_existing:
                seq_dir = PROJECT_ROOT / 'data' / 'clips' / seq
                video_files = list(seq_dir.glob('*.mp4')) + list(seq_dir.glob('*.avi'))
                if not video_files:
                    continue
                clip_ext = video_files[0].suffix
                metric_file = run_dir / seq / f"metric/{seq}_detection.txt"
                if metric_file.exists():
                    print(f"  [{config_name}] {seq}: already done, skipping")
                    continue

            res_dir = run_dir / seq / config_name
            flags = CONFIGS[config_name]

            try:
                stats = run_pipeline(
                    clip_name=seq,
                    result_dir=str(res_dir),
                    config_name=config_name,
                    max_frames=args.max_frames,
                    **flags
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"  [ERROR] {seq}/{config_name}: {e}")
                import traceback
                traceback.print_exc()
                all_stats.append({
                    'config': config_name,
                    'clip': seq,
                    'error': str(e)
                })

    total_time = time.time() - total_start

    # Save summary
    summary_path = run_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'configs': configs_to_run,
            'sequences': sequences_to_run,
            'total_time_s': round(total_time, 1),
            'results': all_stats
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Ablation complete: {total_time:.1f}s total")
    print(f"  Results: {run_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
