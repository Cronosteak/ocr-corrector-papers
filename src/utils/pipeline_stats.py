"""
pipeline_stats.py — Records pipeline timing and volume metrics for the
Dataset Construction section of the paper.
"""

import json
import time
from pathlib import Path
from datetime import datetime

STATS_FILE = Path("data/pipeline_stats.json")


def load_stats() -> dict:
    if STATS_FILE.exists():
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_stats(stats: dict) -> None:
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


class StepTimer:
    """Context manager that times a pipeline step and persists the result."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"\n[{self.step_name}] Starting — {datetime.now().strftime('%H:%M:%S')}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        mins, secs = divmod(int(elapsed), 60)
        print(f"[{self.step_name}] Completed in {mins}m {secs}s")

        stats = load_stats()
        stats[self.step_name] = {
            "duration_seconds": round(elapsed, 1),
            "duration_human": f"{mins}m {secs}s",
            "timestamp": datetime.now().isoformat(),
        }
        save_stats(stats)

    def record(self, key: str, value) -> None:
        """Record an extra value (e.g. n_works, n_pdfs) in the stats."""
        stats = load_stats()
        if self.step_name in stats:
            stats[self.step_name][key] = value
            save_stats(stats)


def print_summary() -> None:
    """Print a summary of all pipeline metrics."""
    stats = load_stats()
    if not stats:
        print("No stats recorded yet.")
        return

    print("\n" + "="*55)
    print("  PIPELINE STATS — for the paper")
    print("="*55)
    for step, data in stats.items():
        print(f"\n  {step}")
        for k, v in data.items():
            if k != "timestamp":
                print(f"    {k}: {v}")
    print("="*55)
