"""
notebooks/run_module.py - Interactive module selector
Run individual pipeline steps interactively.
"""

# Cell 1: Setup
import os
import sys
from pathlib import Path

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

# Cell 2: Status
from utils.checkpoint_utils import print_status
from modules.models.checkpoint_manager import CheckpointManager

print("=== Current Status ===")
print_status()

cm = CheckpointManager(project_root=str(PROJECT_ROOT))
print("\n=== Model Weights ===")
cm.list_available()

# Cell 3: Select step
# EDIT: Set STEP to run
# STEP = 1 -> Extract frames
# STEP = 2 -> Build VOC dataset
# STEP = 3 -> Run inference
# STEP = 4 -> Evaluate results
# STEP = 5 -> Generate report

STEP = 1
CLIP = "clip-01"       # or None for all clips
MAX_FRAMES = None       # limit frames per clip

# Cell 4: Execute
import subprocess
import time

cmd = [sys.executable, "run_module.py", "--step", str(STEP)]
if CLIP:
    cmd.extend(["--clip", CLIP])
if MAX_FRAMES:
    cmd.extend(["--max-frames", str(MAX_FRAMES)])

print(f"Running: {' '.join(cmd)}")
start = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
elapsed = time.time() - start

if result.stdout:
    print(result.stdout)
if result.returncode != 0 and result.stderr:
    print(f"[ERROR]\n{result.stderr[-1000:]}")

print(f"\nElapsed: {elapsed:.1f}s")

# Cell 5: Updated status
print("\n=== Updated Status ===")
print_status()
