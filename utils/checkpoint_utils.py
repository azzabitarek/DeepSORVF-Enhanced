"""
utils/checkpoint_utils.py — Save/load state between Colab sessions.
Tracks progress via a JSON checkpoint file so interrupted work can resume.
"""

import json
import os
from pathlib import Path
from datetime import datetime

CHECKPOINT_FILE = Path(__file__).resolve().parent.parent / "data" / "frames_cache" / "_resume_checkpoint.json"


def load_checkpoint(path=None):
    """Load the resume checkpoint. Returns dict or empty dict if not found."""
    path = Path(path or CHECKPOINT_FILE)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(state, path=None):
    """Save state to the resume checkpoint file."""
    path = Path(path or CHECKPOINT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["_saved_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_step_state(step_name, path=None):
    """Get the state dict for a specific step."""
    cp = load_checkpoint(path)
    return cp.get(step_name, {})


def set_step_state(step_name, step_data, path=None):
    """Update the state for a specific step without touching others."""
    cp = load_checkpoint(path)
    cp[step_name] = step_data
    save_checkpoint(cp, path)


def is_step_complete(step_name, path=None):
    """Check if a step has already been completed."""
    state = get_step_state(step_name, path)
    return state.get("status") == "complete"


def mark_step_complete(step_name, extra=None, path=None):
    """Mark a step as complete with optional extra metadata."""
    state = get_step_state(step_name, path)
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    if extra:
        state.update(extra)
    set_step_state(step_name, state, path)


def mark_step_incomplete(step_name, reason=None, path=None):
    """Mark a step as incomplete (for retry)."""
    state = get_step_state(step_name, path)
    state["status"] = "incomplete"
    state["attempted_at"] = datetime.now().isoformat()
    if reason:
        state["reason"] = reason
    set_step_state(step_name, state, path)


def get_processed_frames(step_name, path=None):
    """Return the set of already-processed frame indices for incremental progress."""
    state = get_step_state(step_name, path)
    return set(state.get("processed_frames", []))


def add_processed_frames(step_name, frame_indices, path=None):
    """Append frame indices to the processed set."""
    state = get_step_state(step_name, path)
    existing = set(state.get("processed_frames", []))
    existing.update(frame_indices)
    state["processed_frames"] = sorted(existing)
    set_step_state(step_name, state, path)


def reset_checkpoint(path=None):
    """Delete the checkpoint file to start fresh."""
    path = Path(path or CHECKPOINT_FILE)
    if path.exists():
        path.unlink()
        print(f"[Checkpoint] Reset: {path}")


def print_status(path=None):
    """Print a human-readable summary of all step states."""
    cp = load_checkpoint(path)
    if not cp:
        print("[Checkpoint] No checkpoint found — starting fresh.")
        return
    print("=" * 60)
    print("  Checkpoint Status")
    print("=" * 60)
    for key, val in cp.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict):
            status = val.get("status", "unknown")
            icon = "[OK]" if status == "complete" else "[..]"
            print(f"  {icon} {key}: {status}")
            if "processed_frames" in val:
                print(f"       frames: {len(val['processed_frames'])}")
    print("=" * 60)
