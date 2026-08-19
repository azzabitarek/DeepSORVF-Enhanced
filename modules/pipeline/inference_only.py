"""
modules/pipeline/inference_only.py — Headless inference pipeline (no GUI, no display).
Processes cached frames or video clips and saves results.
Designed for Colab: saves after each batch, supports resume.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class InferenceRunner:
    """
    Headless inference pipeline — runs detection + tracking + fusion
    without GUI dependencies.

    Parameters
    ----------
    project_root : str or Path
        Path to projet/ directory
    device : str
        'cuda', 'cpu', or 'auto'
    """

    def __init__(self, project_root=None, device="auto"):
        self.project_root = Path(project_root or os.getcwd())
        self.device = device
        self.yolox = None
        self.yolov8 = None
        self.deepsort = None
        self._initialized = False

    def initialize(self):
        """Lazy-load all models (called on first frame or explicitly)."""
        if self._initialized:
            return

        # YOLOX
        from modules.models.yolox_wrapper import YOLOXDetector
        self.yolox = YOLOXDetector(
            model_path=str(self.project_root / "detection_yolox/model_data/YOLOX-final.pth"),
            classes_path=str(self.project_root / "detection_yolox/model_data/ship_classes.txt"),
        )

        # YOLOv8 (optional)
        try:
            from modules.models.yolov8_wrapper import YOLOv8Detector
            self.yolov8 = YOLOv8Detector(
                weights=str(self.project_root / "best.pt"),
                conf=0.30
            )
        except Exception as e:
            print(f'[Pipeline] YOLOv8 not available: {e}')
            self.yolov8 = None

        # DeepSORT
        from deep_sort.utils.parser import get_config
        from deep_sort.deep_sort import DeepSort
        cfg = get_config()
        cfg.merge_from_file(str(self.project_root / "deep_sort/configs/deep_sort.yaml"))
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=(self.device != "cpu"),
            use_reid=True
        )

        self._initialized = True
        print("[Pipeline] All models initialized.")

    def run_on_clip(self, clip_name, result_dir, anti=1, anti_rate=0, max_frames=None):
        """
        Run the full pipeline on a video clip.

        Parameters
        ----------
        clip_name : str
            Name of the clip directory (e.g. 'clip-01')
        result_dir : str or Path
            Where to save results
        anti : int
            Enable anti-occlusion (1=on, 0=off)
        anti_rate : int
            Anti-occlusion rate
        max_frames : int or None
            Limit number of frames to process

        Returns
        -------
        dict with timing and detection statistics
        """
        self.initialize()

        # Import pipeline components
        sys.path.insert(0, str(self.project_root))
        from utils.file_read import read_all, ais_initial, update_time, time2stamp
        from utils.VIS_utils import VISPRO, preprocess_frame
        from utils.AIS_utils import AISPRO
        from utils.FUS_utils import FUSPRO
        from utils.gen_result import gen_result

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Read clip data
        clip_path = str(self.project_root / clip_name) + "/"
        video_path, ais_path, result_video, result_metric, initial_time, camera_para = \
            read_all(clip_path, str(result_dir) + "/")

        cap = cv2.VideoCapture(video_path)
        im_shape = [cap.get(3), cap.get(4)]
        max_dis = min(im_shape) // 2
        fps = int(cap.get(5))
        t = int(1000 / fps)

        # Initialize processors
        AIS = AISPRO(ais_path, ais_initial(ais_path, initial_time)[0], im_shape, t)
        VIS = VISPRO(anti, anti_rate, t)
        FUS = FUSPRO(max_dis, im_shape, t)

        Time = initial_time.copy()
        timestamp0, _ = time2stamp(initial_time)
        times = 0
        time_i = 0
        sum_t = []

        stats = {
            "clip": clip_name,
            "fps": fps,
            "frame_count": 0,
            "total_time_s": 0,
            "avg_fps": 0,
            "detections_per_frame": [],
            "start_time": datetime.now().isoformat()
        }

        print(f"[Pipeline] Processing {clip_name} ({fps} fps, {im_shape[0]}x{im_shape[1]})")

        while True:
            _, im = cap.read()
            if im is None:
                break

            if max_frames and times >= max_frames:
                break

            start = time.time()
            Time, timestamp, Time_name = update_time(Time, t)

            # Process one frame
            AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
            Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis,
                                            pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match']))
            Fus_tra, _ = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)

            end = time.time() - start
            time_i += end

            if timestamp % 1000 < t:
                gen_result(times, Vis_cur, Fus_tra, result_metric, im_shape)
                stats["detections_per_frame"].append(len(Vis_cur))
                times += 1
                sum_t.append(time_i)
                time_i = 0

        cap.release()

        stats["frame_count"] = times
        stats["total_time_s"] = round(sum(sum_t), 4)
        stats["avg_fps"] = round(times / max(sum(sum_t), 0.001), 2)
        stats["end_time"] = datetime.now().isoformat()
        stats["avg_detections"] = round(np.mean(stats["detections_per_frame"]), 2) if stats["detections_per_frame"] else 0

        print(f"[Pipeline] Done: {times} frames, avg {stats['avg_fps']} fps")
        return stats

    def run_on_frame(self, frame_bgr, camera_para, timestamp, AIS_vis, AIS_cur,
                     Vis_tra, bind_inf, anti=1):
        """
        Process a single frame through the pipeline.
        Used for interactive/notebook mode.

        Returns
        -------
        dict with Vis_cur, Fus_tra, detections
        """
        self.initialize()

        from utils.VIS_utils import VISPRO, preprocess_frame
        from utils.AIS_utils import AISPRO
        from utils.FUS_utils import FUSPRO

        t = 33  # ~30fps

        VIS = VISPRO(anti, 0, t)
        Vis_tra, Vis_cur = VIS.feedCap(frame_bgr, timestamp, AIS_vis, bind_inf)

        return {
            "Vis_tra": Vis_tra,
            "Vis_cur": Vis_cur,
            "yolox_detections": len(Vis_cur),
        }
