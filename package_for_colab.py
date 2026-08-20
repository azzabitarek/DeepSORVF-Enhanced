"""
package_for_colab.py — Create a zip package for Google Colab deployment.

Packages:
  - All source code (Python, configs, notebooks)
  - Weights (~121 MB) from weights/
  - setup_colab.ipynb

Excludes:
  - Videos (stay on Google Drive)
  - AIS CSVs (stay on Google Drive)
  - __pycache__, .git, results
  - Training checkpoints (detection_yolox/logs/, checkpoints/)

Usage:
    python package_for_colab.py
    # Output: DeepSORVF_Colab.zip (~135 MB)

Upload to Google Drive, then run setup_colab.ipynb in Colab.
"""
import os
import zipfile
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "DeepSORVF_Colab.zip"

# Directories/files to include
INCLUDE_DIRS = [
    "config",
    "utils",
    "modules",
    "detection_yolox",
    "detection_yolov8",
    "deep_sort",
    "notebooks",
    "weights",
]

INCLUDE_FILES = [
    "run_ablation.py",
    "ablation_runner.py",
    "aggregate_ablation.py",
    "setup_colab.ipynb",
    "phase1_ablation.ipynb",
    "phase2_ablation.ipynb",
    "requirements_colab.txt",
]

# Directories to skip entirely (training checkpoints, logs, etc.)
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "logs",                # training checkpoints (200 × 34 MB = 6.8 GB)
    "checkpoints",         # training checkpoints
}

# Exact file paths (relative to project root) to skip
SKIP_FILES = {
    "deep_sort/deep_sort/deep/checkpoint/ckpt.t7",  # use weights/ckpt.t7
    "detection_yolox/model_data/YOLOX-final.pth",    # use weights/YOLOX-final.pth
    "detection_yolox/model_data/YOLOX-v2-final.pth", # old checkpoint
    "best.pt",                                        # use weights/best.pt
}

def should_skip(path: str) -> bool:
    """Check if a file should be excluded from the package."""
    # Normalize to forward slashes for cross-platform matching
    path = path.replace("\\", "/")
    # Skip exact files
    if path in SKIP_FILES:
        return True
    # Skip by extension
    if path.endswith(".pyc"):
        return True
    # Skip .pth files outside weights/ (training checkpoints)
    if path.endswith(".pth") and not path.startswith("weights/"):
        return True
    return False

def make_package():
    file_count = 0
    total_size = 0

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add directories
        for dirname in INCLUDE_DIRS:
            dirpath = PROJECT_ROOT / dirname
            if not dirpath.exists():
                print(f"  SKIP (not found): {dirname}/")
                continue
            for root, dirs, files in os.walk(dirpath):
                # Skip unwanted subdirectories
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, PROJECT_ROOT).replace("\\", "/")
                    if should_skip(arcname):
                        continue
                    zf.write(fpath, arcname)
                    fsize = os.path.getsize(fpath)
                    total_size += fsize
                    file_count += 1

        # Add individual files
        for fname in INCLUDE_FILES:
            fpath = PROJECT_ROOT / fname
            if fpath.exists():
                zf.write(str(fpath), fname)
                total_size += fpath.stat().st_size
                file_count += 1
            else:
                print(f"  SKIP (not found): {fname}")

    size_mb = total_size / 1_000_000
    print(f"\nPackage created: {OUTPUT}")
    print(f"  Files: {file_count}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"\nUpload to Google Drive:")
    print(f"  1. Copy {OUTPUT.name} to Google Drive root")
    print(f"  2. In Colab: upload and extract, or mount Drive directly")

if __name__ == "__main__":
    print("Creating Colab package...")
    make_package()
