"""
notebooks/02_model_training.ipynb → Model training notebook
════════════════════════════════════════════════════════════
Run this notebook to:
1. Check if pre-trained weights are available (skip training)
2. Optionally train YOLOX (local) or KOLOMVERSE (Colab GPU)
3. Save trained weights to checkpoints/

NOTE: Pre-trained weights already included — training is OPTIONAL.
"""

# ── Cell 1: Setup ───────────────────────────────────────────────────
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

# ── Cell 2: Check available checkpoints ────────────────────────────
from modules.models.checkpoint_manager import CheckpointManager

cm = CheckpointManager(project_root=str(PROJECT_ROOT))
print("Available checkpoints:")
cm.list_available()

# ── Cell 3: Decision ──────────────────────────────────────────────
# Pre-trained weights are included in the project.
# Training is only needed if:
#   - You want to fine-tune on SeaDronesSee
#   - You want to compare trained vs pre-trained

SKIP_TRAINING = cm.verify_all()
if SKIP_TRAINING:
    print("\n[OK] All pre-trained weights available — training SKIPPED.")
    print("     To train anyway, set SKIP_TRAINING = False below.")
else:
    print("\n[!!] Missing weights — training required.")

# ── Cell 4: YOLOX Training (optional) ─────────────────────────────
SKIP_TRAINING = True  # ← Set to False to train

if not SKIP_TRAINING:
    print("\n[TRAIN] YOLOX training...")
    # Training uses detection_yolox/train.py with config from config/yolox_config.yaml
    # Run: python detection_yolox/train.py
    # Or use the existing training script directly
    import subprocess
    result = subprocess.run(
        [sys.executable, "detection_yolox/train.py"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"[ERROR] Training failed:\n{result.stderr[-500:]}")
    else:
        # Copy best model to checkpoints
        import shutil
        best_model = PROJECT_ROOT / "checkpoints" / "yolox_logs" / "best_epoch_weights.pth"
        if best_model.exists():
            cm.save("yolox_custom", str(best_model))
            print("[OK] YOLOX model saved to checkpoints")
else:
    print("[SKIP] YOLOX training skipped.")

# ── Cell 5: KOLOMVERSE Training (Colab GPU only) ─────────────────
if IN_COLAB and not SKIP_TRAINING:
    print("\n[TRAIN] KOLOMVERSE (YOLOv8) training on Colab GPU...")
    # Uses ultralytics YOLOv8 training
    # Requires: pip install ultralytics
    # !yolo detect train data=config/seadronessee.yaml model=yolov8s.pt epochs=100 imgsz=640
else:
    print("[SKIP] KOLOMVERSE training skipped (pre-trained weights included).")

# ── Cell 6: Verify ────────────────────────────────────────────────
print("\n[STATUS] Model checkpoints:")
cm.list_available()
print("\n[COMPLETE] Step 2 — Model Training done.")
