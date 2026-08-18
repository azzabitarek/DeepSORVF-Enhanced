"""
utils/logger.py — Structured logging for pipeline execution.
Supports both console and file output. Resumable between sessions.
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime


class PipelineLogger:
    """
    Logs pipeline events to console + optional CSV/JSON files.
    Each experiment gets its own log directory.
    """

    def __init__(self, log_dir, experiment_name=None, verbose=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.experiment_name = experiment_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        self.csv_path = self.log_dir / f"{self.experiment_name}_log.csv"
        self.summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        self.console_path = self.log_dir / f"{self.experiment_name}_console.txt"

        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._console_lines = []

        header = ["timestamp", "level", "module", "message"]
        if self._csv_file.tell() == 0:
            self._csv_writer.writerow(header)

        self.metrics = {}

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write(self, level, module, message):
        ts = self._ts()
        line = f"[{ts}] [{level}] [{module}] {message}"
        self._csv_writer.writerow([ts, level, module, message])
        self._csv_file.flush()
        self._console_lines.append(line)
        if self.verbose:
            print(line)

    def info(self, module, message):
        self._write("INFO", module, message)

    def warn(self, module, message):
        self._write("WARN", module, message)

    def error(self, module, message):
        self._write("ERROR", module, message)

    def metric(self, key, value, unit=""):
        """Record a numeric metric."""
        self.metrics[key] = {"value": value, "unit": unit, "time": self._ts()}

    def save_summary(self):
        """Write accumulated metrics to JSON."""
        summary = {
            "experiment": self.experiment_name,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "duration_s": (datetime.now() - self.session_start).total_seconds(),
            "metrics": self.metrics,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.info("Logger", f"Summary saved to {self.summary_path}")

    def close(self):
        """Flush and close all log files."""
        self.save_summary()
        with open(self.console_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._console_lines))
        self._csv_file.close()
        self.info("Logger", "Session complete.")
