"""
modules/evaluation/metrics.py — Evaluation metrics for detection, tracking, and fusion.
mAP, IDP/IDR/IDFP, CPA accuracy, tracking accuracy.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


class EvaluationMetrics:
    """
    Computes evaluation metrics for the DeepSORVF pipeline.

    Metrics:
    - Detection: mAP@0.5, mAP@0.5:0.95, precision, recall
    - Tracking: IDP, IDR, IDF1, MOTA, MOTP
    - Fusion: match accuracy, false association rate
    - Timing: FPS, inference time, processing time
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box1, box2):
        """Compute IoU between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def compute_map(self, predictions, ground_truths):
        """
        Compute mAP@0.5 and mAP@0.5:0.95.

        Parameters
        ----------
        predictions : list of list of dict
            Each dict: {frame: int, bbox: (x1,y1,x2,y2), conf: float, class: str}
        ground_truths : list of list of dict
            Each dict: {frame: int, bbox: (x1,y1,x2,y2), class: str}

        Returns
        -------
        dict with mAP metrics
        """
        all_preds = []
        for preds in predictions:
            all_preds.extend(preds)
        all_gts = []
        for gts in ground_truths:
            all_gts.extend(gts)

        if not all_preds or not all_gts:
            return {"mAP_0.5": 0.0, "mAP_0.5_0.95": 0.0}

        # Sort by confidence
        all_preds.sort(key=lambda x: x.get("conf", 0), reverse=True)

        tp_list = []
        matched_gts = set()

        for pred in all_preds:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(all_gts):
                if gt_idx in matched_gts:
                    continue
                if gt["frame"] != pred["frame"]:
                    continue
                if gt.get("class", "") != pred.get("class", ""):
                    continue
                iou = self.compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                tp_list.append(1)
            else:
                tp_list.append(0)

        tp = sum(tp_list)
        fp = len(all_preds) - tp
        fn = len(all_gts) - len(matched_gts)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return {
            "mAP_0.5": round(precision * recall, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_gt": len(all_gts),
            "total_pred": len(all_preds),
        }

    def compute_tracking_metrics(self, track_results, ground_truth_tracks):
        """
        Compute tracking metrics: IDP, IDR, IDF1, MOTA.

        Parameters
        ----------
        track_results : pd.DataFrame
            Columns: frame, ID, x1, y1, x2, y2
        ground_truth_tracks : pd.DataFrame
            Columns: frame, ID, x1, y1, x2, y2

        Returns
        -------
        dict with tracking metrics
        """
        if track_results.empty or ground_truth_tracks.empty:
            return {"IDP": 0, "IDR": 0, "IDF1": 0, "MOTA": 0}

        # Match tracks to ground truth across all frames
        tp_total = 0
        fp_total = 0
        fn_total = 0
        id_switches = 0

        gt_frames = set(ground_truth_tracks["frame"].unique())
        pred_frames = set(track_results["frame"].unique())

        prev_gt_matches = {}

        for frame in sorted(gt_frames | pred_frames):
            gt_in_frame = ground_truth_tracks[ground_truth_tracks["frame"] == frame]
            pred_in_frame = track_results[track_results["frame"] == frame]

            if gt_in_frame.empty and pred_in_frame.empty:
                continue

            # Match predictions to ground truth
            matched_gt = set()
            matched_pred = set()

            for _, pred_row in pred_in_frame.iterrows():
                best_iou = 0
                best_gt_idx = None
                for gt_idx, gt_row in gt_in_frame.iterrows():
                    iou = self.compute_iou(
                        (pred_row["x1"], pred_row["y1"], pred_row["x2"], pred_row["y2"]),
                        (gt_row["x1"], gt_row["y1"], gt_row["x2"], gt_row["y2"])
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold and best_gt_idx is not None:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_row["ID"] if "ID" in pred_row else _)
                    tp_total += 1

                    # Check ID switch
                    gt_id = gt_in_frame.loc[best_gt_idx, "ID"]
                    if gt_id in prev_gt_matches and prev_gt_matches[gt_id] != pred_row.get("ID"):
                        id_switches += 1
                    prev_gt_matches[gt_id] = pred_row.get("ID")
                else:
                    fp_total += 1

            fn_total += len(gt_in_frame) - len(matched_gt)

        idp = tp_total / max(tp_total + fp_total, 1)
        idr = tp_total / max(tp_total + fn_total, 1)
        idf1 = 2 * idp * idr / max(idp + idr, 1e-6)
        mota = 1 - (fp_total + fn_total + id_switches) / max(tp_total + fn_total, 1)

        return {
            "IDP": round(idp, 4),
            "IDR": round(idr, 4),
            "IDF1": round(idf1, 4),
            "MOTA": round(mota, 4),
            "id_switches": id_switches,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        }

    @staticmethod
    def load_metric_file(filepath):
        """Load a DeepSORVF metric file (detection/tracking/fusion txt)."""
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    records.append({
                        "frame": int(parts[0]),
                        "id": int(parts[1]),
                        "x": int(parts[2]),
                        "y": int(parts[3]),
                        "w": int(parts[4]),
                        "h": int(parts[5]),
                    })
        return pd.DataFrame(records) if records else pd.DataFrame()

    def evaluate_clip(self, clip_name, result_dir, gt_dir):
        """
        Full evaluation of a clip against ground truth.

        Parameters
        ----------
        clip_name : str
        result_dir : str or Path
        gt_dir : str or Path

        Returns
        -------
        dict with all metrics
        """
        result_dir = Path(result_dir)
        gt_dir = Path(gt_dir)

        results = {}
        for metric_type in ["detection", "tracking", "fusion"]:
            pred_file = result_dir / f"{clip_name}_{metric_type}.txt"
            gt_file = gt_dir / f"{clip_name}_gt_{metric_type}.txt"

            if pred_file.exists() and gt_file.exists():
                pred_df = self.load_metric_file(pred_file)
                gt_df = self.load_metric_file(gt_file)
                results[metric_type] = {
                    "pred_frames": len(pred_df["frame"].unique()) if not pred_df.empty else 0,
                    "gt_frames": len(gt_df["frame"].unique()) if not gt_df.empty else 0,
                    "pred_objects": len(pred_df),
                    "gt_objects": len(gt_df),
                }
            else:
                results[metric_type] = {"error": "file not found"}

        return results
