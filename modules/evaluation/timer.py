"""
modules/evaluation/timer.py — Profiling and FPS measurement.
Tracks inference time, preprocessing time, and overall pipeline FPS.
"""

import time
import numpy as np
from collections import defaultdict


class TimerProfiler:
    """
    Per-stage timer for pipeline profiling.

    Usage:
        profiler = TimerProfiler()
        profiler.start("detection")
        # ... run detection ...
        profiler.stop("detection")

        print(profiler.summary())
    """

    def __init__(self):
        self.stages = defaultdict(list)
        self._active = {}

    def start(self, stage_name):
        """Start timing a stage."""
        self._active[stage_name] = time.perf_counter()

    def stop(self, stage_name):
        """Stop timing a stage and record the duration."""
        if stage_name not in self._active:
            return
        elapsed = time.perf_counter() - self._active.pop(stage_name)
        self.stages[stage_name].append(elapsed)

    def get_stats(self, stage_name):
        """Get timing statistics for a stage."""
        times = self.stages.get(stage_name, [])
        if not times:
            return {"count": 0, "total_ms": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        times_ms = [t * 1000 for t in times]
        return {
            "count": len(times_ms),
            "total_ms": round(sum(times_ms), 2),
            "avg_ms": round(np.mean(times_ms), 2),
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
            "std_ms": round(np.std(times_ms), 2) if len(times_ms) > 1 else 0,
        }

    def summary(self):
        """Print a formatted summary of all stages."""
        lines = ["=" * 60, "  Timer Profiler Summary", "=" * 60]
        total_all = 0
        for stage, times in sorted(self.stages.items()):
            stats = self.get_stats(stage)
            total_all += stats["total_ms"]
            lines.append(f"  {stage:<25} {stats['count']:>5} calls  "
                        f"avg={stats['avg_ms']:>8.2f}ms  "
                        f"total={stats['total_ms']:>8.2f}ms")
        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':<25} {'':>5}       "
                    f"{'':>8}        total={total_all:>8.2f}ms")
        lines.append("=" * 60)

        # FPS
        if total_all > 0:
            avg_per_frame = total_all / sum(len(v) for v in self.stages.values())
            lines.append(f"  Estimated FPS: {1000 / avg_per_frame:.2f}")

        return "\n".join(lines)

    def to_dict(self):
        """Export all stats as a dictionary."""
        return {stage: self.get_stats(stage) for stage in self.stages}

    def reset(self):
        """Clear all recorded timings."""
        self.stages.clear()
        self._active.clear()
