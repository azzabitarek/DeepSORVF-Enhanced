"""
run_ablation.py — Headless pipeline runner for ablation study.

Usage:
    python run_ablation.py --clip clip-01 --config C5 --result-dir ./ablation_results/

This script runs the full DeepSORVF pipeline WITHOUT any GUI (no cv2.imshow,
no video writing). It is designed to be called by ablation_runner.py for
batch execution of multiple configurations.
"""
import os
import sys
import time
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline(clip_name, result_dir,
                 use_ensemble=True, use_static_filter=True,
                 ais_enabled=True, anti=1,
                 use_dtw=True, use_angle_penalty=True,
                 use_binding=True, use_hungarian=True,
                 max_frames=None, config_name='default'):
    """
    Run the DeepSORVF pipeline headless on a single clip.

    Parameters
    ----------
    clip_name : str
        Name of the clip directory (e.g. 'clip-01')
    result_dir : str or Path
        Where to write output files
    use_ensemble : bool
        Enable KOLOMVERSE maritime detections
    use_static_filter : bool
        Enable static structure filter (wind farm, shoreline)
    ais_enabled : bool
        Enable AIS data processing
    anti : int
        OAR anti-occlusion mode (0=off, 1=on)
    use_dtw : bool
        Use DTW for trajectory cost (False = Euclidean distance)
    use_angle_penalty : bool
        Use angular penalty in DTW and heading_penalty
    use_binding : bool
        Enable binding (consecutive match locking)
    use_hungarian : bool
        Use Hungarian algorithm (False = greedy assignment)
    max_frames : int or None
        Maximum frames to process (None = all)
    config_name : str
        Name of this configuration (for logging)

    Returns
    -------
    dict with timing statistics
    """
    from utils.file_read import read_all, ais_initial, update_time
    from utils.AIS_utils import AISPRO
    from utils.VIS_utils import VISPRO
    from utils.FUS_utils import FUSPRO
    from utils.gen_result import gen_result

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Read clip data
    clip_path = str(PROJECT_ROOT / 'data' / 'clips' / clip_name) + "/"
    video_path, ais_path, result_video, result_metric, initial_time, camera_para = \
        read_all(clip_path, str(result_dir) + "/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    im_shape = [cap.get(3), cap.get(4)]
    t = int(1000 / fps)
    max_dis = min(im_shape) // 2

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Initialize AIS
    ais_file, timestamp0, time0 = ais_initial(ais_path, initial_time)
    AIS = AISPRO(ais_path, ais_file, im_shape, t)

    # Initialize VIS with ablation flags
    VIS = VISPRO(anti=anti, val=0, t=t, ais_enabled=ais_enabled)

    # Initialize FUS with ablation flags
    FUS = FUSPRO(max_dis, im_shape, t,
                 use_dtw=use_dtw,
                 use_angle_penalty=use_angle_penalty,
                 use_binding=use_binding,
                 use_hungarian=use_hungarian)

    # Enable ablation logging if requested
    ablation_log_path = result_dir / f"{clip_name}_{config_name}_ablation_log.csv"
    FUS._ablation_log_path = str(ablation_log_path)

    Time = initial_time.copy()
    bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    # Timing
    total_time = 0
    frame_count = 0
    detection_seconds = 0
    start_all = time.time()

    # DeepSORT reinit support (prevents STATUS_STACK_BUFFER_OVERRUN native crash
    # from state accumulation on Windows).
    import torch
    import utils.VIS_utils as vis_module
    from deep_sort.utils.parser import get_config as ds_get_config
    from deep_sort.deep_sort import DeepSort as DS_DeepSort
    ds_cfg = ds_get_config()
    ds_cfg.merge_from_file(str(PROJECT_ROOT / "deep_sort/configs/deep_sort.yaml"))

    def _reinit_processors():
        """Recreate deepsort + AIS/VIS/FUS to prevent Windows native crash."""
        nonlocal AIS, VIS, FUS
        vis_module.deepsort = DS_DeepSort(
            str(PROJECT_ROOT / ds_cfg.DEEPSORT.REID_CKPT),
            max_dist=ds_cfg.DEEPSORT.MAX_DIST,
            min_confidence=ds_cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=ds_cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=ds_cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=ds_cfg.DEEPSORT.MAX_AGE,
            n_init=ds_cfg.DEEPSORT.N_INIT,
            nn_budget=ds_cfg.DEEPSORT.NN_BUDGET,
            use_cuda=torch.cuda.is_available(),
            use_reid=False
        )
        ais_file_fresh, _, _ = ais_initial(ais_path, initial_time)
        AIS = AISPRO(ais_path, ais_file_fresh, im_shape, t)
        VIS = VISPRO(anti=anti, val=0, t=t, ais_enabled=ais_enabled)
        FUS = FUSPRO(max_dis, im_shape, t,
                     use_dtw=use_dtw,
                     use_angle_penalty=use_angle_penalty,
                     use_binding=use_binding,
                     use_hungarian=use_hungarian)
        FUS._ablation_log_path = str(ablation_log_path)

    # Reinit every REINIT_FRAMES to prevent native crash.
    # At 25 frames = 1 detection second (fps=25), deepsort sees exactly 1
    # detection second before reset. With tentative tracks patch, vis_cur
    # still has data even without n_init confirmation.
    REINIT_FRAMES = 25
    frames_since_reinit = 0

    while True:
        ret, im = cap.read()
        if not ret:
            break

        frame_idx = frame_count
        if max_frames and frame_idx >= max_frames:
            break

        # Reinit all processors periodically to prevent native crash
        frames_since_reinit += 1
        if frames_since_reinit >= REINIT_FRAMES:
            _reinit_processors()
            frames_since_reinit = 0

        start = time.time()
        Time, timestamp, Time_name = update_time(Time, t)

        # Pipeline stages
        AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
        Vis_tra, Vis_cur = VIS.feedCap(
            im, timestamp, AIS_vis, bin_inf,
            ais_enabled=ais_enabled,
            use_ensemble=use_ensemble,
            use_static_filter=use_static_filter
        )
        if ais_enabled:
            Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
        else:
            Fus_tra = pd.DataFrame()

        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        if timestamp % 1000 < t:
            gen_result(detection_seconds, Vis_cur, Fus_tra, result_metric, im_shape)
            detection_seconds += 1

    cap.release()
    wall_time = time.time() - start_all

    stats = {
        'config': config_name,
        'clip': clip_name,
        'total_frames': frame_count,
        'detection_seconds': detection_seconds,
        'wall_time_s': round(wall_time, 2),
        'avg_ms_per_frame': round(total_time / max(frame_count, 1) * 1000, 1),
    }

    # Write stats
    with open(result_dir / f"{clip_name}_{config_name}_stats.json", 'w') as f:
        import json
        json.dump(stats, f, indent=2)

    print(f"  [{config_name}] {clip_name}: {frame_count} frames, "
          f"{detection_seconds} det-sec, {wall_time:.1f}s wall, "
          f"{stats['avg_ms_per_frame']:.1f} ms/frame")

    return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run single ablation config')
    parser.add_argument('--clip', required=True)
    parser.add_argument('--config', default='C5')
    parser.add_argument('--result-dir', default='./ablation_results/')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--use-ensemble', type=int, default=1)
    parser.add_argument('--use-static-filter', type=int, default=1)
    parser.add_argument('--ais-enabled', type=int, default=1)
    parser.add_argument('--anti', type=int, default=1)
    parser.add_argument('--use-dtw', type=int, default=1)
    parser.add_argument('--use-angle-penalty', type=int, default=1)
    parser.add_argument('--use-binding', type=int, default=1)
    parser.add_argument('--use-hungarian', type=int, default=1)
    args = parser.parse_args()

    run_pipeline(
        clip_name=args.clip,
        result_dir=args.result_dir,
        config_name=args.config,
        use_ensemble=bool(args.use_ensemble),
        use_static_filter=bool(args.use_static_filter),
        ais_enabled=bool(args.ais_enabled),
        anti=args.anti,
        use_dtw=bool(args.use_dtw),
        use_angle_penalty=bool(args.use_angle_penalty),
        use_binding=bool(args.use_binding),
        use_hungarian=bool(args.use_hungarian),
        max_frames=args.max_frames,
    )
