"""
notebooks/01_dataset_preparation.ipynb → Dataset preparation notebook
═══════════════════════════════════════════════════════════════════════
Run this notebook to:
1. Mount Google Drive
2. Extract frames from video clips
3. Cache frames for later annotation
4. Convert COCO annotations (SeaDronesSee) to YOLO format

Steps: 1 → 2 → cache → Drive sync
"""

# ── Cell 1: Setup ───────────────────────────────────────────────────
import os
import sys
from pathlib import Path

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("[OK] Google Drive mounted")
except ImportError:
    IN_COLAB = False
    print("[--] Not in Colab — local mode")

# Project root
if IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/DeepSORVF_Project/projet")
else:
    PROJECT_ROOT = Path(os.getcwd())

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))
print(f"[OK] Project root: {PROJECT_ROOT}")

# ── Cell 2: Verify data ────────────────────────────────────────────
from utils.drive_utils import list_clips, get_clip_path

clips = list_clips()
print(f"Available clips: {clips}")

for clip in clips:
    clip_path = get_clip_path(clip)
    videos = list(clip_path.glob("*.avi")) + list(clip_path.glob("*.mp4"))
    cam = clip_path / "camera_para.txt"
    gt_dir = clip_path / "gt"
    print(f"  {clip}: video={len(videos)}, camera={'OK' if cam.exists() else 'MISSING'}, "
          f"gt={'OK' if gt_dir.exists() else 'MISSING'}")

# ── Cell 3: Extract frames ─────────────────────────────────────────
from modules.dataset.extract_frames import FrameExtractor
from utils.file_read import read_all

TARGET_FRAMES = 140  # per clip

for clip_name in clips:
    clip_path = get_clip_path(clip_name)
    videos = list(clip_path.glob("*.avi")) + list(clip_path.glob("*.mp4"))
    if not videos:
        continue

    # Read camera params
    cam_file = clip_path / "camera_para.txt"
    with open(cam_file, "r") as f:
        cam_line = f.readlines()[0][1:-2]
        camera_para = list(map(float, cam_line.split(",")))

    output_dir = PROJECT_ROOT / "data" / "frames_cache" / clip_name
    if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) >= TARGET_FRAMES:
        print(f"[SKIP] {clip_name}: already extracted ({len(list(output_dir.glob('*.jpg')))} frames)")
        continue

    print(f"\n[EXTRACT] {clip_name}...")
    extractor = FrameExtractor(videos[0], camera_para, output_dir, target_frames=TARGET_FRAMES)
    frames = extractor.extract()
    print(f"  Extracted {len(frames)} frames to {output_dir}")

# ── Cell 4: Cache management ───────────────────────────────────────
from modules.dataset.cache_frames import FrameCache

for clip_name in clips:
    cache_dir = PROJECT_ROOT / "data" / "frames_cache"
    cache = FrameCache(str(cache_dir), clip_name)
    info = cache.get_cache_info()
    print(f"  {clip_name}: {info}")

# ── Cell 5: Convert SeaDronesSee (if available) ───────────────────
seadronesee_path = PROJECT_ROOT / "data" / "processed" / "seaDronesSee"
if seadronesee_path.exists():
    from modules.dataset.convert_coco import COCOToYOLO

    coco_json = None
    for jf in seadronesee_path.rglob("*.json"):
        if "instances" in jf.name or "annotations" in jf.name:
            coco_json = jf
            break

    if coco_json:
        print(f"\n[CONVERT] SeaDronesSee: {coco_json}")
        converter = COCOToYOLO(
            coco_json_path=str(coco_json),
            images_dir=str(seadronesee_path / "images"),
            output_labels_dir=str(PROJECT_ROOT / "data" / "processed" / "seaDronesSee" / "labels")
        )
        stats = converter.convert()
        converter.generate_data_yaml(
            str(PROJECT_ROOT / "config" / "seadronessee.yaml")
        )
        print(f"  Stats: {stats}")
else:
    print("[--] SeaDronesSee dataset not found — skipping COCO conversion")

# ── Cell 6: Sync to Drive ──────────────────────────────────────────
if IN_COLAB:
    from utils.drive_utils import sync_to_drive
    sync_to_drive(
        str(PROJECT_ROOT / "data" / "frames_cache"),
        str(PROJECT_ROOT / "data" / "frames_cache").replace("/content/drive/MyDrive/DeepSORVF_Project/projet", "/content/drive/MyDrive/DeepSORVF_Project/data/frames_cache")
    )
    print("[OK] Frames synced to Drive")

print("\n[COMPLETE] Step 1 — Dataset Preparation done.")
