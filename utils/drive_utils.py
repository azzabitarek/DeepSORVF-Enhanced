"""
utils/drive_utils.py — Google Drive mount, path resolution, symlink management.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive(mount_point="/content/drive"):
    """Mount Google Drive (no-op if not Colab or already mounted)."""
    if not is_colab():
        print("[Drive] Not in Colab — skipping mount.")
        return
    if os.path.exists(mount_point):
        print(f"[Drive] Already mounted at {mount_point}")
        return
    from google.colab import drive
    drive.mount(mount_point)
    print(f"[Drive] Mounted at {mount_point}")


def get_drive_root():
    """Return the DeepSORVF root on Google Drive."""
    return Path(os.environ.get(
        "DRIVE_ROOT",
        "/content/drive/MyDrive/DeepSORVF_Project"
    ))


def sync_to_drive(src_dir, drive_dir=None, overwrite=False):
    """
    Copy local project data to Google Drive for persistence.
    Called at end of a Colab session to save progress.
    """
    drive_dir = Path(drive_dir or get_drive_root())
    drive_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_dir)
    if not src.exists():
        print(f"[Drive] Source {src} does not exist — skipping sync.")
        return
    dest = drive_dir / src.name
    if dest.exists() and not overwrite:
        print(f"[Drive] {dest} exists — use overwrite=True to replace.")
        return
    if src.is_dir():
        shutil.copytree(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))
    print(f"[Drive] Synced {src} → {dest}")


def sync_from_drive(drive_dir, local_dir=None):
    """Copy data FROM Google Drive to local project."""
    drive_dir = Path(drive_dir)
    local_dir = Path(local_dir or PROJECT_ROOT)
    if not drive_dir.exists():
        print(f"[Drive] {drive_dir} does not exist.")
        return
    dest = local_dir / drive_dir.name
    if drive_dir.is_dir():
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(drive_dir), str(dest))
    else:
        shutil.copy2(str(drive_dir), str(dest))
    print(f"[Drive] Synced {drive_dir} → {dest}")


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_clip_path(clip_name):
    """Resolve a clip directory path by name."""
    clips_dir = PROJECT_ROOT / 'data' / 'clips'
    clips = {
        "clip-01": clips_dir / "clip-01",
        "Video-29": clips_dir / "Video-29",
        "Video-34": clips_dir / "Video-34",
        "Video-28": clips_dir / "Video-28",
        "Video-10": clips_dir / "Video-10",
        "clip-02": clips_dir / "clip-02",
        "clip-10": clips_dir / "clip-10",
    }
    return clips.get(clip_name, clips_dir / clip_name)


def list_clips():
    """List all available clip directories."""
    clips = []
    for d in PROJECT_ROOT.iterdir():
        if d.is_dir() and (d / "camera_para.txt").exists():
            clips.append(d.name)
    return sorted(clips)


def setup_project_symlinks(drive_root=None):
    """
    Create symlinks from local project to Drive data.
    Useful when code lives locally but data is on Drive.
    """
    drive_root = Path(drive_root or get_drive_root())
    if not drive_root.exists():
        print(f"[Drive] {drive_root} not found — skipping symlinks.")
        return

    links = {
        "data/frames_cache": drive_root / "data/frames_cache",
        "data/processed": drive_root / "data/processed",
        "data/results": drive_root / "data/results",
        "checkpoints": drive_root / "checkpoints",
    }
    for local_rel, drive_path in links.items():
        local_path = PROJECT_ROOT / local_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists() and drive_path.exists():
            os.symlink(str(drive_path), str(local_path))
            print(f"[Drive] Linked {local_path} → {drive_path}")
