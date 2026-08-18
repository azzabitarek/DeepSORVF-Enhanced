"""
run_module.py — Command-line selector for modular pipeline execution.

Usage:
    python run_module.py --step 1                   # Extract frames
    python run_module.py --step 2                   # Build VOC dataset
    python run_module.py --step 3                   # Run inference
    python run_module.py --step 4                   # Evaluate results
    python run_module.py --step 5                   # Generate report
    python run_module.py --step 1 --clip clip-01    # Specific clip
    python run_module.py --status                   # Show checkpoint status
    python run_module.py --reset                    # Reset all progress
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_extract_frames(args):
    """Step 1: Extract frames from video clips."""
    from modules.dataset.extract_frames import FrameExtractor
    from modules.dataset.cache_frames import FrameCache
    from utils.drive_utils import get_clip_path, list_clips

    clips = [args.clip] if args.clip else list_clips()
    print(f"[Step 1] Extracting frames from: {clips}")

    for clip_name in clips:
        clip_path = get_clip_path(clip_name)
        video_files = list(clip_path.glob("*.avi")) + list(clip_path.glob("*.mp4"))
        if not video_files:
            print(f"  No video found in {clip_path} — skipping.")
            continue

        # Read camera parameters
        cam_file = clip_path / "camera_para.txt"
        if cam_file.exists():
            with open(cam_file, "r") as f:
                cam_line = f.readlines()[0][1:-2]
                camera_para = list(map(float, cam_line.split(",")))
        else:
            print(f"  No camera_para.txt in {clip_path} — skipping.")
            continue

        output_dir = PROJECT_ROOT / "data" / "frames_cache" / clip_name
        extractor = FrameExtractor(video_files[0], camera_para, output_dir, target_frames=args.target_frames)
        frames = extractor.extract()

        # Cache
        cache = FrameCache(str(PROJECT_ROOT / "data" / "frames_cache"), clip_name)
        # Save just metadata (frames are saved as JPEG in output_dir)
        meta = {"frame_count": len(frames), "clip": clip_name}
        cache.save_progress("extraction", list(range(len(frames))))

    print("[Step 1] Frame extraction complete.")


def step_build_dataset(args):
    """Step 2: Build VOC dataset and convert to YOLO format."""
    from modules.dataset.build_voc import VOCBuilder

    voc_root = PROJECT_ROOT / "data" / "processed" / "VOCdevkit"
    classes_path = PROJECT_ROOT / "detection_yolox" / "model_data" / "ship_classes.txt"

    builder = VOCBuilder(str(voc_root), str(classes_path))

    # Check if annotations exist
    ann_count = len(list((voc_root / "VOC2007" / "Annotations").glob("*.xml")))
    if ann_count == 0:
        print("[Step 2] No annotations found. Please annotate frames first.")
        print(f"  Expected location: {voc_root / 'VOC2007' / 'Annotations'}")
        return

    print(f"[Step 2] Building dataset from {ann_count} annotations...")

    # Build splits
    stats = builder.build_splits(
        trainval_pct=args.trainval_pct,
        train_pct=args.train_pct
    )

    # Convert to YOLO
    yolo_dir = PROJECT_ROOT / "data" / "processed" / "yolo_labels"
    yolo_stats = builder.convert_to_yolo(str(yolo_dir))

    print(f"[Step 2] Dataset built: {stats}")
    print(f"[Step 2] YOLO labels → {yolo_dir}")


def step_inference(args):
    """Step 3: Run inference pipeline."""
    from modules.pipeline.inference_only import InferenceRunner
    from modules.pipeline.batch_runner import BatchRunner

    result_dir = PROJECT_ROOT / "data" / "results" / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pipeline = InferenceRunner(project_root=str(PROJECT_ROOT))
    runner = BatchRunner(pipeline, batch_size=args.batch_size, checkpoint_name=f"inference_{args.clip or 'all'}")

    if args.clip:
        clips = [args.clip]
    else:
        from utils.drive_utils import list_clips
        clips = list_clips()

    for clip_name in clips:
        print(f"\n[Step 3] Running inference on {clip_name}...")
        stats = runner.run_clip_batched(
            clip_name,
            str(result_dir),
            max_frames=args.max_frames,
            resume=not args.fresh
        )
        # Save clip stats
        clip_result = result_dir / f"{clip_name}_stats.json"
        with open(clip_result, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n[Step 3] Results → {result_dir}")


def step_evaluate(args):
    """Step 4: Evaluate results against ground truth."""
    from modules.evaluation.metrics import EvaluationMetrics

    metrics = EvaluationMetrics(iou_threshold=args.iou_threshold)
    result_dir = PROJECT_ROOT / "data" / "results"

    if args.clip:
        clips = [args.clip]
    else:
        from utils.drive_utils import list_clips
        clips = list_clips()

    all_results = {}
    for clip_name in clips:
        clip_path = PROJECT_ROOT / clip_name
        gt_dir = clip_path / "gt"
        eval_result = metrics.evaluate_clip(clip_name, result_dir, gt_dir)
        all_results[clip_name] = eval_result
        print(f"[Step 4] {clip_name}: {eval_result}")

    # Save evaluation
    eval_path = result_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Step 4] Evaluation saved → {eval_path}")


def step_report(args):
    """Step 5: Generate experiment report."""
    from modules.evaluation.report_generator import ReportGenerator

    result_dir = PROJECT_ROOT / "data" / "results"
    report = ReportGenerator(str(result_dir), experiment_name=args.experiment_name)

    # Load existing results
    for clip_dir in result_dir.iterdir():
        if clip_dir.suffix == ".json" and "stats" in clip_dir.name:
            clip_name = clip_dir.stem.replace("_stats", "")
            with open(clip_dir) as f:
                data = json.load(f)
            report.add_clip_result(clip_name, data)

    report.generate_all()
    print(f"[Step 5] Reports generated in {result_dir}")


def show_status(args):
    """Show checkpoint status."""
    from utils.checkpoint_utils import print_status, load_checkpoint
    from modules.models.checkpoint_manager import CheckpointManager

    print("\n=== Checkpoint Status ===")
    print_status()

    print("\n=== Model Weights ===")
    cm = CheckpointManager(project_root=str(PROJECT_ROOT))
    cm.list_available()


def reset_progress(args):
    """Reset all progress checkpoints."""
    from utils.checkpoint_utils import reset_checkpoint
    reset_checkpoint()
    print("[Reset] All progress cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSORVF Modular Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_module.py --step 1 --clip clip-01
  python run_module.py --step 3 --clip Video-29 --max-frames 50
  python run_module.py --status
  python run_module.py --reset
        """
    )

    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                       help="Pipeline step to execute")
    parser.add_argument("--clip", type=str, default=None,
                       help="Process a specific clip (e.g. clip-01)")
    parser.add_argument("--status", action="store_true",
                       help="Show checkpoint status")
    parser.add_argument("--reset", action="store_true",
                       help="Reset all progress")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh (ignore checkpoints)")

    # Step 1 options
    parser.add_argument("--target-frames", type=int, default=140,
                       help="Target frames per clip for extraction")

    # Step 2 options
    parser.add_argument("--trainval-pct", type=float, default=0.95)
    parser.add_argument("--train-pct", type=float, default=0.95)

    # Step 3 options
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)

    # Step 4 options
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    # Step 5 options
    parser.add_argument("--experiment-name", type=str, default=None)

    args = parser.parse_args()

    if args.status:
        show_status(args)
        return
    if args.reset:
        reset_progress(args)
        return
    if args.step is None:
        parser.print_help()
        return

    start = time.time()
    steps = {
        1: ("Extract Frames", step_extract_frames),
        2: ("Build Dataset", step_build_dataset),
        3: ("Run Inference", step_inference),
        4: ("Evaluate Results", step_evaluate),
        5: ("Generate Report", step_report),
    }

    name, func = steps[args.step]
    print(f"\n{'='*60}")
    print(f"  Step {args.step}: {name}")
    print(f"{'='*60}\n")

    func(args)

    elapsed = time.time() - start
    print(f"\n[Done] Step {args.step} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
