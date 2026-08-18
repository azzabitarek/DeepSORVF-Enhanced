"""
notebooks/03_pipeline_runtime.ipynb → Pipeline execution notebook
═══════════════════════════════════════════════════════════════════
Run this notebook to:
1. Load models (YOLOX + YOLOv8 + DeepSORT)
2. Run inference pipeline on test clips
3. Save detection/tracking/fusion results
4. Profile timing (FPS measurement)

Steps: load models → run clips → save results → sync Drive
"""

# ── Cell 1: Setup ───────────────────────────────────────────────────
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/DeepSORVF_Project/projet")
else:
    PROJECT_ROOT = Path(os.getcwd())

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# ── Cell 2: Initialize pipeline ────────────────────────────────────
from modules.pipeline.inference_only import InferenceRunner
from modules.evaluation.timer import TimerProfiler
from utils.drive_utils import list_clips

profiler = TimerProfiler()
pipeline = InferenceRunner(project_root=str(PROJECT_ROOT))

print("Initializing models...")
profiler.start("model_load")
pipeline.initialize()
profiler.stop("model_load")
print(profiler.summary())

# ── Cell 3: Run inference on clips ─────────────────────────────────
CLIPS_TO_RUN = ["clip-01", "Video-29", "Video-34"]  # ← Edit this list
MAX_FRAMES = None  # ← Set to limit frames (e.g., 50)
EXPERIMENT_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

result_dir = PROJECT_ROOT / "data" / "results" / EXPERIMENT_NAME
result_dir.mkdir(parents=True, exist_ok=True)

all_results = {}

for clip_name in CLIPS_TO_RUN:
    print(f"\n{'='*60}")
    print(f"  Processing: {clip_name}")
    print(f"{'='*60}")

    profiler.start(f"clip_{clip_name}")
    stats = pipeline.run_on_clip(
        clip_name,
        str(result_dir),
        max_frames=MAX_FRAMES
    )
    profiler.stop(f"clip_{clip_name}")

    all_results[clip_name] = stats
    print(f"  FPS: {stats['avg_fps']}, Frames: {stats['frame_count']}, "
          f"Avg detections: {stats['avg_detections']}")

    # Save per-clip results
    with open(result_dir / f"{clip_name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

# ── Cell 4: Timing summary ────────────────────────────────────────
print("\n" + profiler.summary())

# Save profiler results
with open(result_dir / "profiling.json", "w") as f:
    json.dump(profiler.to_dict(), f, indent=2)

# ── Cell 5: Save experiment config ────────────────────────────────
experiment_config = {
    "name": EXPERIMENT_NAME,
    "created_at": datetime.now().isoformat(),
    "clips": CLIPS_TO_RUN,
    "max_frames": MAX_FRAMES,
    "results_dir": str(result_dir),
    "profiling": profiler.to_dict(),
}

with open(result_dir / "experiment_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

# ── Cell 6: Sync to Drive ──────────────────────────────────────────
if IN_COLAB:
    from utils.drive_utils import sync_to_drive
    sync_to_drive(str(result_dir))
    print("[OK] Results synced to Drive")

print(f"\n[COMPLETE] Step 3 — Pipeline Runtime done.")
print(f"Results: {result_dir}")
