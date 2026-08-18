"""
modules/evaluation/report_generator.py — Generate experiment reports.
Supports Markdown, JSON, and CSV output.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """
    Generates structured experiment reports from pipeline results.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save reports
    experiment_name : str
        Name/ID for this experiment
    """

    def __init__(self, output_dir, experiment_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.results = {}

    def add_clip_result(self, clip_name, timing_stats, detection_stats=None,
                        tracking_stats=None, fusion_stats=None):
        """Add results for a processed clip."""
        self.results[clip_name] = {
            "timing": timing_stats,
            "detection": detection_stats or {},
            "tracking": tracking_stats or {},
            "fusion": fusion_stats or {},
        }

    def add_comparison(self, model_name, metrics):
        """Add model comparison data."""
        if "_comparisons" not in self.results:
            self.results["_comparisons"] = {}
        self.results["_comparisons"][model_name] = metrics

    def generate_markdown(self, title="DeepSORVF Experiment Report"):
        """Generate a Markdown report."""
        lines = [
            f"# {title}",
            f"",
            f"**Experiment:** {self.experiment_name}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "---",
            "",
        ]

        # Clip results
        for clip_name, data in sorted(self.results.items()):
            if clip_name.startswith("_"):
                continue
            lines.append(f"## {clip_name}")
            lines.append("")

            timing = data.get("timing", {})
            if timing:
                lines.append("### Timing")
                lines.append(f"- Frames processed: {timing.get('frame_count', 'N/A')}")
                lines.append(f"- Average FPS: {timing.get('avg_fps', 'N/A')}")
                lines.append(f"- Total time: {timing.get('total_time_s', 'N/A')}s")
                lines.append(f"- Avg detections/frame: {timing.get('avg_detections', 'N/A')}")
                lines.append("")

            for metric_type in ["detection", "tracking", "fusion"]:
                metric_data = data.get(metric_type, {})
                if metric_data and "error" not in metric_data:
                    lines.append(f"### {metric_type.capitalize()} Results")
                    for k, v in metric_data.items():
                        lines.append(f"- {k}: {v}")
                    lines.append("")

        # Model comparisons
        comparisons = self.results.get("_comparisons", {})
        if comparisons:
            lines.append("## Model Comparison")
            lines.append("")
            for model_name, metrics in comparisons.items():
                lines.append(f"### {model_name}")
                for k, v in metrics.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

        md_path = self.output_dir / f"{self.experiment_name}_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Report] Markdown → {md_path}")
        return md_path

    def generate_json(self):
        """Generate a JSON report."""
        report = {
            "experiment": self.experiment_name,
            "generated_at": datetime.now().isoformat(),
            "results": self.results,
        }
        json_path = self.output_dir / f"{self.experiment_name}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Report] JSON → {json_path}")
        return json_path

    def generate_csv(self):
        """Generate a CSV summary table."""
        rows = []
        for clip_name, data in sorted(self.results.items()):
            if clip_name.startswith("_"):
                continue
            timing = data.get("timing", {})
            rows.append({
                "clip": clip_name,
                "frames": timing.get("frame_count", ""),
                "avg_fps": timing.get("avg_fps", ""),
                "total_time_s": timing.get("total_time_s", ""),
                "avg_detections": timing.get("avg_detections", ""),
            })

        if not rows:
            return None

        csv_path = self.output_dir / f"{self.experiment_name}_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Report] CSV → {csv_path}")
        return csv_path

    def generate_all(self):
        """Generate all report formats."""
        self.generate_markdown()
        self.generate_json()
        self.generate_csv()
