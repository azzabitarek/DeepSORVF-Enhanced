"""
modules/models/checkpoint_manager.py — Centralized checkpoint management.
Loads/saves model weights, handles Drive sync, supports resume.
"""

import os
import shutil
import torch
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    Manages all model checkpoints for the project.

    Checks Drive first, then local. Handles copy-back for persistence.
    """

    DEFAULT_CHECKPOINTS = {
        "yolox_pretrained": "./detection_yolox/model_data/YOLOX-final.pth",
        "yolox_custom": "./checkpoints/yolox_smd_final.pth",
        "kolomverse_pretrained": "./best.pt",
        "kolomverse_custom": "./checkpoints/kolomverse_final.pt",
        "deepsort_reid": "./deep_sort/deep_sort/deep/checkpoint/ckpt.t7",
    }

    def __init__(self, project_root=None, drive_root=None):
        self.project_root = Path(project_root or os.getcwd())
        self.drive_root = Path(drive_root) if drive_root else None
        self.checkpoints = {}
        self._discover()

    def _discover(self):
        """Find all available checkpoints."""
        for name, rel_path in self.DEFAULT_CHECKPOINTS.items():
            local = self.project_root / rel_path
            if local.exists():
                self.checkpoints[name] = local

    def get(self, name, prefer_drive=True):
        """
        Get checkpoint path by name. Checks Drive first if available.

        Returns
        -------
        Path or None
        """
        if name in self.checkpoints:
            return self.checkpoints[name]
        # Check Drive
        if prefer_drive and self.drive_root:
            for rel in self.DEFAULT_CHECKPOINTS.values():
                drive_path = self.drive_root / rel
                if drive_path.exists():
                    self.checkpoints[name] = drive_path
                    return drive_path
        return None

    def save(self, name, source_path, copy_to_drive=True):
        """
        Save a checkpoint to the project and optionally to Drive.

        Parameters
        ----------
        name : str
            Checkpoint name (e.g. 'yolox_custom')
        source_path : str or Path
            Path to the trained weights file
        copy_to_drive : bool
            Also copy to Drive for persistence
        """
        source = Path(source_path)
        if not source.exists():
            print(f"[Checkpoint] Source {source} not found.")
            return

        # Determine local destination
        rel = self.DEFAULT_CHECKPOINTS.get(name, f"checkpoints/{name}.pth")
        local_dest = self.project_root / rel
        local_dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(source), str(local_dest))
        self.checkpoints[name] = local_dest
        print(f"[Checkpoint] Saved {name} → {local_dest}")

        # Copy to Drive
        if copy_to_drive and self.drive_root:
            drive_dest = self.drive_root / rel
            drive_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(drive_dest))
            print(f"[Checkpoint] Also saved to Drive → {drive_dest}")

    def list_available(self):
        """Print and return all available checkpoints."""
        available = {}
        for name in self.DEFAULT_CHECKPOINTS:
            path = self.get(name)
            status = "OK" if path else "MISSING"
            size = ""
            if path and path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                size = f" ({size_mb:.1f} MB)"
            print(f"  [{status}] {name}: {path or 'not found'}{size}")
            available[name] = {"path": str(path), "available": path is not None}
        return available

    def verify_all(self):
        """Check that all required checkpoints are available."""
        required = ["yolox_pretrained", "deepsort_reid"]
        missing = [r for r in required if self.get(r) is None]
        if missing:
            print(f"[Checkpoint] MISSING required: {missing}")
            return False
        print("[Checkpoint] All required checkpoints available.")
        return True

    def load_weights(self, name, model=None, map_location=None):
        """
        Load checkpoint weights into a PyTorch model.

        Returns
        -------
        dict of state_dict or the model with loaded weights
        """
        path = self.get(name)
        if path is None:
            raise FileNotFoundError(f"Checkpoint '{name}' not found.")
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(str(path), map_location=map_location)
        if model is not None:
            model.load_state_dict(state_dict)
            return model
        return state_dict
