"""
utils/detection_logger.py
─────────────────────────────────────────────────────────────────────────────
Per-frame detection logger for the dual-model setup:
  - YOLOX        → vessel detections (tracked + fused with AIS)
  - KOLOMVERSE   → maritime object detections (display only)

Outputs two files in the result folder:
  detections_log.csv   — one row per detection per frame
  detections_summary.txt — human-readable session summary

CSV columns:
  timestamp | frame | model | class | confidence | x1 | y1 | x2 | y2 | suppressed_by_yolox
"""

import os
import csv
import time
from datetime import datetime


class DetectionLogger:
    CSV_COLUMNS = [
        'timestamp', 'frame', 'time_name',
        'model', 'class', 'confidence',
        'x1', 'y1', 'x2', 'y2',
        'suppressed_by_yolox'
    ]

    def __init__(self, result_dir, clip_name):
        os.makedirs(result_dir, exist_ok=True)
        self.csv_path     = os.path.join(result_dir, f'{clip_name}_detections_log.csv')
        self.summary_path = os.path.join(result_dir, f'{clip_name}_detections_summary.txt')
        self.clip_name    = clip_name
        self.session_start = datetime.now()

        # Open CSV and write header
        self._f   = open(self.csv_path, 'w', newline='')
        self._csv = csv.DictWriter(self._f, fieldnames=self.CSV_COLUMNS)
        self._csv.writeheader()

        # In-memory counters for summary
        self.frame_count       = 0
        self.yolox_total       = 0
        self.kolomverse_total  = 0
        self.kolomverse_suppressed = 0
        self.yolox_classes     = {}   # {class_name: count}
        self.kolo_classes      = {}   # {class_name: count}

        print(f'[Logger] Logging to {self.csv_path}')

    # ── Main logging call — call this once per detection frame ────────────────
    def log_frame(self, timestamp, frame_idx, time_name,
                  yolox_bboxes, maritime_bboxes_kept, maritime_bboxes_suppressed):
        """
        Parameters
        ----------
        yolox_bboxes              : list of (x1,y1,x2,y2,cls,conf) from YOLOX
        maritime_bboxes_kept      : list of (x1,y1,x2,y2,cls,conf) from KOLOMVERSE
                                    that passed the IoU suppression check
        maritime_bboxes_suppressed: list of (x1,y1,x2,y2,cls,conf) from KOLOMVERSE
                                    that were suppressed because YOLOX already covered them
        """
        self.frame_count += 1

        # — YOLOX detections ──────────────────────────────────────────────────
        for (x1, y1, x2, y2, cls, conf) in yolox_bboxes:
            conf_val = float(conf)
            self._csv.writerow({
                'timestamp':          timestamp,
                'frame':              frame_idx,
                'time_name':          time_name,
                'model':              'YOLOX',
                'class':              cls,
                'confidence':         f'{conf_val:.4f}',
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'suppressed_by_yolox': 'N/A'
            })
            self.yolox_total += 1
            self.yolox_classes[cls] = self.yolox_classes.get(cls, 0) + 1

        # — KOLOMVERSE kept ───────────────────────────────────────────────────
        for (x1, y1, x2, y2, cls, conf) in maritime_bboxes_kept:
            conf_val = float(conf)
            self._csv.writerow({
                'timestamp':          timestamp,
                'frame':              frame_idx,
                'time_name':          time_name,
                'model':              'KOLOMVERSE',
                'class':              cls,
                'confidence':         f'{conf_val:.4f}',
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'suppressed_by_yolox': 'No'
            })
            self.kolomverse_total += 1
            self.kolo_classes[cls] = self.kolo_classes.get(cls, 0) + 1

        # — KOLOMVERSE suppressed (YOLOX overlap) ─────────────────────────────
        for (x1, y1, x2, y2, cls, conf) in maritime_bboxes_suppressed:
            conf_val = float(conf)
            self._csv.writerow({
                'timestamp':          timestamp,
                'frame':              frame_idx,
                'time_name':          time_name,
                'model':              'KOLOMVERSE',
                'class':              cls,
                'confidence':         f'{conf_val:.4f}',
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'suppressed_by_yolox': 'Yes'
            })
            self.kolomverse_total += 1
            self.kolomverse_suppressed += 1
            self.kolo_classes[cls] = self.kolo_classes.get(cls, 0) + 1

        # Flush every frame — guarantees data on disk even if crash
        self._f.flush()

    # ── Call at end of video ──────────────────────────────────────────────────
    def close(self):
        self._f.flush()
        self._f.close()
        self._write_summary()
        print(f'[Logger] Saved → {self.csv_path}')
        print(f'[Logger] Saved → {self.summary_path}')

    def _write_summary(self):
        duration = datetime.now() - self.session_start
        kolo_kept = self.kolomverse_total - self.kolomverse_suppressed

        lines = [
            '═' * 60,
            f'  DeepSORVF Dual-Model Detection Log — {self.clip_name}',
            '═' * 60,
            f'  Session start  : {self.session_start.strftime("%Y-%m-%d %H:%M:%S")}',
            f'  Duration       : {str(duration).split(".")[0]}',
            f'  Detection frames logged: {self.frame_count}',
            '',
            '── YOLOX (vessels → tracking + AIS fusion) ─────────────',
            f'  Total detections : {self.yolox_total}',
            f'  Avg per frame    : {self.yolox_total/max(1,self.frame_count):.2f}',
            f'  Classes detected :',
        ]
        for cls, cnt in sorted(self.yolox_classes.items(), key=lambda x: -x[1]):
            lines.append(f'      {cls:<25} {cnt}')

        lines += [
            '',
            '── KOLOMVERSE (maritime objects → display only) ─────────',
            f'  Total detections : {self.kolomverse_total}',
            f'  Kept (shown)     : {kolo_kept}',
            f'  Suppressed by YOLOX overlap: {self.kolomverse_suppressed}',
            f'  Avg kept per frame: {kolo_kept/max(1,self.frame_count):.2f}',
            f'  Classes detected :',
        ]
        if self.kolo_classes:
            for cls, cnt in sorted(self.kolo_classes.items(), key=lambda x: -x[1]):
                lines.append(f'      {cls:<25} {cnt}')
        else:
            lines.append('      (none — no maritime objects detected in this video)')
            lines.append('      → Try a video with kayaks, zodiacs, buoys, or sailboats')
            lines.append('        to trigger KOLOMVERSE detections.')

        lines += [
            '',
            f'  CSV log : {self.csv_path}',
            '═' * 60,
        ]

        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        # Also print to console
        print('\n' + '\n'.join(lines))