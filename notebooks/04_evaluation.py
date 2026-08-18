"""
notebooks/04_evaluation.ipynb → Evaluation notebook
═════════════════════════════════════════════════════
Run this notebook to:
1. Load results from Step 3
2. Compare against ground truth
3. Compute detection, tracking, and fusion metrics
4. Generate comparison tables
"""

# ── Cell 1: Setup ───────────────────────────────────────────────────
import os
import sys
import json
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

# ── Cell 2: Load results ──────────────────────────────────────────
from utils.drive_utils import list_clips

results_dir = PROJECT_ROOT / "data" / "results"
experiments = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")])
print("Available experiments:")
for exp in experiments:
    print(f"  {exp.name}")

if not experiments:
    print("[!!] No experiment results found. Run Step 3 first.")
else:
    LATEST_EXP = experiments[-1]
    print(f"\nUsing: {LATEST_EXP.name}")

# ── Cell 3: Evaluate against ground truth ──────────────────────────
from modules.evaluation.metrics import EvaluationMetrics

evaluator = EvaluationMetrics(iou_threshold=0.5)
clips = list_clips()

evaluation_results = {}
for clip_name in clips:
    gt_dir = PROJECT_ROOT / clip_name / "gt"
    if not gt_dir.exists():
        continue

    result = evaluator.evaluate_clip(clip_name, LATEST_EXP, gt_dir)
    evaluation_results[clip_name] = result
    print(f"\n{clip_name}:")
    for metric_type, data in result.items():
        print(f"  {metric_type}: {data}")

# Save
eval_path = LATEST_EXP / "evaluation_results.json"
with open(eval_path, "w") as f:
    json.dump(evaluation_results, f, indent=2)
print(f"\n[OK] Evaluation saved → {eval_path}")

# ── Cell 4: Compare models ────────────────────────────────────────
# If multiple experiments exist, compare them
if len(experiments) >= 2:
    print("\n=== Model Comparison ===")
    for exp in experiments[-3:]:  # Last 3 experiments
        eval_file = exp / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            print(f"\n{exp.name}:")
            for clip, metrics in data.items():
                print(f"  {clip}: {metrics}")

# ── Cell 5: Generate report ───────────────────────────────────────
from modules.evaluation.report_generator import ReportGenerator

report = ReportGenerator(
    str(LATEST_EXP),
    experiment_name=LATEST_EXP.name
)

# Add clip results
for clip_name in clips:
    stats_file = LATEST_EXP / f"{clip_name}_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            data = json.load(f)
        report.add_clip_result(clip_name, data)

# Add evaluation
report.add_clip_result("evaluation", evaluation_results)

report.generate_all()
print(f"\n[COMPLETE] Step 4 — Evaluation done.")
print(f"Reports in: {LATEST_EXP}")
