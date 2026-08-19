"""
modules/pipeline/batch_runner.py — Batch processing with checkpoint resume.
Splits large jobs into chunks, saves progress after each chunk.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.checkpoint_utils import (
    get_step_state, set_step_state, add_processed_frames,
    get_processed_frames
)


class BatchRunner:
    """
    Runs the inference pipeline in batches with checkpoint resume.

    Parameters
    ----------
    pipeline : InferenceRunner
        Initialized inference pipeline
    batch_size : int
        Number of frames per batch
    checkpoint_name : str
        Name for checkpoint tracking
    """

    def __init__(self, pipeline, batch_size=20, checkpoint_name="inference"):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.checkpoint_name = checkpoint_name

    def run_clip_batched(self, clip_name, result_dir, max_frames=None, resume=True):
        """
        Process a video clip in batches with resume support.

        Parameters
        ----------
        clip_name : str
            Clip directory name
        result_dir : str or Path
            Output directory
        max_frames : int or None
            Maximum frames to process
        resume : bool
            Resume from last checkpoint

        Returns
        -------
        dict with overall statistics
        """
        self.pipeline.initialize()

        # Import needed modules
        project_root = self.pipeline.project_root
        sys_path_orig = sys.path.copy()
        sys.path.insert(0, str(project_root))

        from utils.file_read import read_all, ais_initial, update_time, time2stamp

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        clip_path = str(project_root / clip_name) + "/"
        video_path, ais_path, result_video, result_metric, initial_time, camera_para = \
            read_all(clip_path, str(result_dir) + "/")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        im_shape = [cap.get(3), cap.get(4)]

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Load resume state
        processed = set()
        if resume:
            processed = get_processed_frames(self.checkpoint_name)

        # Calculate batches
        all_indices = list(range(total_frames))
        remaining = [i for i in all_indices if i not in processed]
        batches = [remaining[i:i+self.batch_size] for i in range(0, len(remaining), self.batch_size)]

        print(f"[Batch] {clip_name}: {total_frames} frames total, {len(remaining)} remaining, {len(batches)} batches")

        overall_stats = {
            "clip": clip_name,
            "total_frames": total_frames,
            "already_processed": len(processed),
            "batches_total": len(batches),
            "batches_completed": 0,
            "start_time": datetime.now().isoformat(),
        }

        try:
            for batch_idx, batch_frames in enumerate(batches):
                print(f"[Batch] {batch_idx+1}/{len(batches)}: frames {batch_frames[0]}-{batch_frames[-1]}")

                # Process this batch
                batch_stats = self._process_batch(
                    cap, batch_frames, initial_time, fps, im_shape,
                    camera_para, ais_path, result_metric, project_root
                )

                # Update checkpoint
                add_processed_frames(self.checkpoint_name, batch_frames)
                overall_stats["batches_completed"] = batch_idx + 1

                # Save batch results
                batch_result_path = result_dir / f"batch_{batch_idx:03d}_stats.json"
                with open(batch_result_path, "w") as f:
                    json.dump(batch_stats, f, indent=2)

                print(f"[Batch] Done: {batch_stats['frame_count']} frames, "
                      f"avg {batch_stats['avg_ms_per_frame']:.1f} ms/frame")

        finally:
            cap.release()
            overall_stats["end_time"] = datetime.now().isoformat()
            sys.path = sys_path_orig

        print(f"[Batch] Complete: {overall_stats['batches_completed']}/{len(batches)} batches")
        return overall_stats

    def _process_batch(self, cap, frame_indices, initial_time, fps, im_shape,
                       camera_para, ais_path, result_metric, project_root):
        """Process a single batch of frames sequentially (no cap.set)."""
        import sys
        sys.path.insert(0, str(project_root))
        from utils.file_read import update_time
        from utils.VIS_utils import VISPRO
        from utils.AIS_utils import AISPRO
        from utils.FUS_utils import FUSPRO
        from utils.gen_result import gen_result
        import torch
        from deep_sort.utils.parser import get_config
        from deep_sort.deep_sort import DeepSort
        import utils.VIS_utils as vis_module

        t = int(1000 / fps)
        AIS = AISPRO(ais_path, [], im_shape, t)
        VIS = VISPRO(1, 0, t)
        FUS = FUSPRO(min(im_shape) // 2, im_shape, t)

        # Reinitialize global deepsort to avoid state accumulation across batches
        cfg = get_config()
        cfg.merge_from_file(str(project_root / "deep_sort/configs/deep_sort.yaml"))
        vis_module.deepsort = DeepSort(
            str(project_root / cfg.DEEPSORT.REID_CKPT),
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=torch.cuda.is_available(),
            use_reid=True
        )

        batch_stats = {
            "frame_indices": frame_indices,
            "frame_count": 0,
            "total_ms": 0,
            "detections": [],
        }

        output_idx = 0

        for expected_idx in frame_indices:
            ret, im = cap.read()
            if not ret:
                continue

            Time = initial_time.copy()
            for _ in range(expected_idx):
                Time, _, _ = update_time(Time, t)

            start = time.time()
            Time, timestamp, Time_name = update_time(Time, t)

            AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
            Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis,
                                            pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match']))
            Fus_tra, _ = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)

            elapsed_ms = (time.time() - start) * 1000
            batch_stats["total_ms"] += elapsed_ms
            batch_stats["detections"].append(len(Vis_cur))
            batch_stats["frame_count"] += 1

            if timestamp % 1000 < t:
                gen_result(output_idx, Vis_cur, Fus_tra, result_metric, im_shape)
                output_idx += 1

        batch_stats["avg_ms_per_frame"] = round(
            batch_stats["total_ms"] / max(batch_stats["frame_count"], 1), 2
        )
        return batch_stats

    def get_progress(self):
        """Get current processing progress."""
        processed = get_processed_frames(self.checkpoint_name)
        state = get_step_state(self.checkpoint_name)
        return {
            "frames_processed": len(processed),
            "status": state.get("status", "in_progress"),
            "last_update": state.get("updated_at", "unknown"),
        }
