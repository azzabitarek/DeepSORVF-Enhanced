"""
modules/dataset/extract_frames.py — Intelligent frame extraction from SMD video clips.
Selects 140 representative frames (diverse lighting, weather, traffic density)
and saves JPEG + metadata for later annotation.
"""

import os
import cv2
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime


class FrameExtractor:
    """
    Extracts representative frames from a video clip.

    Selection strategy:
    - Uniform temporal sampling as baseline
    - Brightness-variance scoring for lighting diversity
    - Edge-density scoring for scene complexity
    - Histogram divergence from already-selected frames

    Parameters
    ----------
    video_path : str or Path
        Path to the .avi/.mp4 video file
    camera_para : list
        Camera parameters [lon, lat, hdir, vdir, h, FOV_h, FOV_v, fx, fy, u0, v0]
    output_dir : str or Path
        Where to save extracted frames (JPEG) + metadata
    target_frames : int
        Number of frames to extract (default 140)
    """

    def __init__(self, video_path, camera_para, output_dir, target_frames=140):
        self.video_path = Path(video_path)
        self.camera_para = camera_para
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_frames = target_frames
        self.metadata = {
            "video_path": str(self.video_path),
            "camera_para": camera_para,
            "extraction_time": None,
            "frames": []
        }

    def _score_frame(self, frame):
        """Score a frame for diversity: combines brightness, edge density, and complexity."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        return {
            "brightness": brightness,
            "edge_density": edge_density,
            "entropy": entropy,
            "score": 0.4 * edge_density + 0.3 * entropy + 0.3 * (brightness / 255.0)
        }

    def _hist_divergence(self, hist1, hist2):
        """Jensen-Shannon divergence between two histograms."""
        hist1 = hist1.astype(np.float64) + 1e-10
        hist2 = hist2.astype(np.float64) + 1e-10
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
        m = 0.5 * (hist1 + hist2)
        kl1 = np.sum(hist1 * np.log2(hist1 / m))
        kl2 = np.sum(hist2 * np.log2(hist2 / m))
        return float(0.5 * (kl1 + kl2))

    def extract(self, resume_frame=0):
        """
        Run frame extraction. Returns list of frame metadata dicts.

        Parameters
        ----------
        resume_frame : int
            Start from this frame index (for session resume).
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Uniform sample positions
        sample_indices = np.linspace(0, total_frames - 1, self.target_frames * 3, dtype=int)

        # Pass 1: score all candidate frames
        candidates = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            scores = self._score_frame(frame)
            candidates.append((int(idx), frame, scores))

        # Pass 2: greedy diversity selection
        selected = []
        selected_hists = []
        candidates.sort(key=lambda x: x[2]["score"], reverse=True)

        for idx, frame, scores in candidates:
            if len(selected) >= self.target_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()

            # Check diversity from already-selected frames
            if selected_hists:
                min_div = min(self._hist_divergence(hist, sh) for sh in selected_hists)
                if min_div < 0.05 and len(selected) > self.target_frames // 2:
                    continue

            # Save frame
            frame_name = f"frame_{len(selected):04d}.jpg"
            frame_path = self.output_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_meta = {
                "index": len(selected),
                "original_frame": int(idx),
                "filename": frame_name,
                "timestamp_ms": int(idx * 1000 / fps),
                "timestamp_sec": round(idx / fps, 3),
                "brightness": scores["brightness"],
                "edge_density": scores["edge_density"],
                "entropy": scores["entropy"],
                "width": W,
                "height": H,
            }
            selected.append(frame_meta)
            selected_hists.append(hist)

        cap.release()

        self.metadata["frames"] = selected
        self.metadata["extraction_time"] = datetime.now().isoformat()
        self.metadata["video_info"] = {
            "total_frames": total_frames,
            "fps": fps,
            "width": W,
            "height": H,
            "duration_s": round(total_frames / fps, 2)
        }

        # Save metadata
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"[Extractor] {len(selected)} frames extracted → {self.output_dir}")
        print(f"[Extractor] Metadata → {meta_path}")
        return selected

    @staticmethod
    def load_metadata(output_dir):
        """Load previously saved metadata."""
        meta_path = Path(output_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
