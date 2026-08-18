"""
modules/dataset/cache_frames.py — Frame cache management for session resume.
Saves/loads frame sets + metadata as pickle files so Colab sessions
can pick up where they left off.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime


class FrameCache:
    """
    Manages cached frames and metadata between Colab sessions.

    Features:
    - Save/load frame sets with metadata
    - Track which frames have been annotated
    - Resume from last checkpoint
    - Clean cache to free space
    """

    def __init__(self, cache_dir, clip_name=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clip_name = clip_name or "default"
        self.cache_file = self.cache_dir / f"{self.clip_name}_frames.pkl"
        self.progress_file = self.cache_dir / f"{self.clip_name}_progress.json"

    def save_frames(self, frames, metadata=None):
        """
        Save frames + metadata to cache.

        Parameters
        ----------
        frames : list of numpy arrays (BGR images)
        metadata : dict with frame info
        """
        cache_data = {
            "clip_name": self.clip_name,
            "saved_at": datetime.now().isoformat(),
            "frame_count": len(frames),
            "frames": frames,
            "metadata": metadata or {}
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Cache] Saved {len(frames)} frames → {self.cache_file}")
        print(f"[Cache] Size: {os.path.getsize(self.cache_file) / 1024 / 1024:.1f} MB")

    def load_frames(self):
        """Load cached frames. Returns (frames, metadata) or (None, None)."""
        if not self.cache_file.exists():
            print(f"[Cache] No cache found at {self.cache_file}")
            return None, None
        with open(self.cache_file, "rb") as f:
            cache_data = pickle.load(f)
        print(f"[Cache] Loaded {cache_data['frame_count']} frames from {self.cache_file}")
        return cache_data["frames"], cache_data.get("metadata", {})

    def save_progress(self, step_name, processed_indices):
        """Save annotation progress for a specific step."""
        progress = self._load_progress()
        progress[step_name] = {
            "processed": sorted(processed_indices),
            "count": len(processed_indices),
            "updated_at": datetime.now().isoformat()
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def load_progress(self, step_name):
        """Load annotation progress for a step. Returns set of processed indices."""
        progress = self._load_progress()
        step_data = progress.get(step_name, {})
        return set(step_data.get("processed", []))

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def clean(self, keep_metadata=True):
        """Remove cached frames to free disk space."""
        if self.cache_file.exists():
            os.remove(self.cache_file)
            print(f"[Cache] Deleted {self.cache_file}")
        if not keep_metadata and self.progress_file.exists():
            os.remove(self.progress_file)
            print(f"[Cache] Deleted {self.progress_file}")

    def get_cache_info(self):
        """Return cache statistics."""
        info = {
            "cache_exists": self.cache_file.exists(),
            "clip_name": self.clip_name,
        }
        if self.cache_file.exists():
            info["size_mb"] = round(os.path.getsize(self.cache_file) / 1024 / 1024, 2)
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            info["frame_count"] = data["frame_count"]
            info["saved_at"] = data.get("saved_at", "unknown")
        progress = self._load_progress()
        info["steps"] = {k: v["count"] for k, v in progress.items()}
        return info
