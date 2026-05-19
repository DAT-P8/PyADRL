"""Live inspection of a Ray Tune experiment for the gridworld tuner.

Usage:
    python check_tuner.py experiments/tuner/260518_1138
    python check_tuner.py experiments/tuner/260518_1138/gridworld_tune
    python check_tuner.py experiments/tuner/260518_1138 --metric score_p
    python check_tuner.py experiments/tuner/260518_1138 --window 10

What it does:
  * Scans every trial directory under {path}/gridworld_tune (auto-resolves
    if you pass either the timestamped parent or the gridworld_tune subdir).
  * For each trial: reads result.json (JSONL — one report per line), pulls
    the latest reported metrics, computes the mean of the selection metric
    over the last `window` reports, and classifies trial status.
  * Prints a leaderboard sorted by the selection metric.
  * Flags contamination (full_capture_rate + breach_rate > 1.0), NaN values,
    and trials missing the new composite scores (runs that predate the
    score_p / comb_score implementation).

Safe to run while the tuner is writing: each result.json is opened
read-only and a partially-flushed trailing line is skipped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

WINDOW_DEFAULT = 5
# After this many seconds with no new report, a trial is shown as IDLE.
# Picked to be larger than a typical eval_interval cycle (a few minutes per
# tune.report call) so transient gaps aren't flagged.
IDLE_THRESHOLD_SEC = 300
TOL = 1e-3
TRIAL_DIR_GLOBS = ("iterative_trainable_*", "alternate_trainable_*")


def find_gridworld_tune_dir(path: Path) -> Path | None:
    if path.name == "gridworld_tune" and path.is_dir():
        return path
    candidate = path / "gridworld_tune"
    if candidate.is_dir():
        return candidate
    return None


def iter_trial_dirs(root: Path):
    for pat in TRIAL_DIR_GLOBS:
        yield from root.glob(pat)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, tolerating an incomplete trailing line."""
    entries: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Last line being flushed mid-write — skip silently.
                    pass
    except FileNotFoundError:
        pass
    return entries


def safe_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def fmt_age(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s ago"
    if sec < 3600:
        return f"{int(sec / 60)}m ago"
    return f"{sec / 3600:.1f}h ago"


def fmt(v: float | None, width: int = 8, prec: int = 3) -> str:
    if v is None:
        return f"{'—':<{width}}"
    return f"{v:<{width}.{prec}f}"


def classify_status(last: dict, now: float) -> str:
    if last.get("done") is True:
        return "DONE"
    ts = safe_float(last, "timestamp")
    if ts is None:
        return "UNKNOWN"
    age = now - ts
    if age <= IDLE_THRESHOLD_SEC:
        return f"ACTIVE ({fmt_age(age)})"
    return f"IDLE ({fmt_age(age)})"


def summarize_trial(trial_dir: Path, window: int, metric: str, now: float) -> dict:
    entries = read_jsonl(trial_dir / "result.json")
    summary = {
        "trial_id": trial_dir.name,
        "iters": 0,
        "metric_mean": None,
        "latest": {},
        "status": "EMPTY",
        "contaminated": 0,
        "nan_count": 0,
        "missing_metric": True,
    }
    if not entries:
        return summary

    last = entries[-1]
    summary["trial_id"] = last.get("trial_id", trial_dir.name)
    # algo_iteration is the actual algo.train() count (emitted by the
    # rebased trainables). Falls back to Ray's training_iteration for
    # runs that predate that change.
    summary["iters"] = (
        last.get("algo_iteration") or last.get("training_iteration") or len(entries)
    )
    summary["status"] = classify_status(last, now)

    interesting = (
        "comb_score",
        "score_p",
        "capture_score",
        "weighted_acs",
        "full_capture_rate",
        "breach_rate",
        "mean_reward",
        "pursuer_success",
    )
    for k in interesting:
        v = safe_float(last, k)
        if v is not None:
            summary["latest"][k] = v
    summary["missing_metric"] = metric not in summary["latest"]

    # Mean of selection metric over the trailing window. Matches the
    # stability-window approach get_best_n uses in gridworld_tuner.py
    # (more reliable than peak-single-eval given RL eval noise).
    vals = [safe_float(e, metric) for e in entries[-window:]]
    vals = [v for v in vals if v is not None]
    if vals:
        summary["metric_mean"] = sum(vals) / len(vals)

    # Whole-history sanity counters.
    for e in entries:
        fc = safe_float(e, "full_capture_rate")
        br = safe_float(e, "breach_rate")
        if fc is not None and br is not None and fc + br > 1.0 + TOL:
            summary["contaminated"] += 1
        for k in ("comb_score", "score_p", "mean_reward", "breach_rate"):
            raw = e.get(k)
            if isinstance(raw, float) and raw != raw:
                summary["nan_count"] += 1

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "path",
        type=Path,
        help="experiments/tuner/{timestamp} or its gridworld_tune subdir",
    )
    ap.add_argument(
        "--metric",
        default="comb_score",
        help="metric to rank trials by (default: comb_score; falls back gracefully if absent)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=WINDOW_DEFAULT,
        help=f"average the metric over the last N reports (default: {WINDOW_DEFAULT})",
    )
    args = ap.parse_args()

    root = find_gridworld_tune_dir(args.path)
    if root is None:
        print(
            f"ERROR: no gridworld_tune directory at {args.path} "
            "(looked for ./gridworld_tune too)"
        )
        return 1

    trial_dirs = sorted(iter_trial_dirs(root))
    if not trial_dirs:
        print(f"ERROR: no trial directories under {root}")
        return 1

    now = time.time()
    summaries = [summarize_trial(d, args.window, args.metric, now) for d in trial_dirs]
    # Sort by metric_mean desc; None values land at the bottom.
    summaries.sort(key=lambda s: (s["metric_mean"] is None, -(s["metric_mean"] or 0.0)))

    n_total = len(summaries)
    n_reporting = sum(1 for s in summaries if s["iters"] > 0)
    n_done = sum(1 for s in summaries if s["status"].startswith("DONE"))
    n_active = sum(1 for s in summaries if s["status"].startswith("ACTIVE"))
    n_idle = sum(1 for s in summaries if s["status"].startswith("IDLE"))
    n_empty = sum(1 for s in summaries if s["iters"] == 0)
    n_contam = sum(s["contaminated"] for s in summaries)
    n_nan = sum(s["nan_count"] for s in summaries)
    n_missing_metric = sum(
        1 for s in summaries if s["iters"] > 0 and s["missing_metric"]
    )

    print(f"\n=== {root} ===")
    print(
        f"Trials: {n_total} total — "
        f"{n_active} active, {n_idle} idle, {n_done} done, {n_empty} no reports"
    )
    print(
        f"Ranking by mean of '{args.metric}' over last {args.window} reports."
    )
    if n_missing_metric:
        print(
            f"NOTE: {n_missing_metric}/{n_reporting} reporting trials have no "
            f"'{args.metric}' field (run predates composite-score impl — "
            "try --metric mean_reward or pursuer_success)."
        )
    print()

    # Header label for the trailing-window-mean column. The line above
    # already names the metric in full; keep this column compact.
    mean_label = f"avg({args.window})"
    header = (
        f"{'trial':<14} {'iter':>5}  "
        f"{mean_label:<12}"
        f"{'comb':<8}{'score_p':<8}{'capture':<8}{'breach':<8}"
        f"{'mean_rew':<10} status"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        L = s["latest"]
        # Prefer the new composite capture metric; fall back to legacy
        # full_capture_rate so old result.json files still show something.
        capture_display = L.get("capture_score")
        if capture_display is None:
            capture_display = L.get("full_capture_rate")
        print(
            f"{s['trial_id']:<14} {s['iters']:>5}  "
            f"{fmt(s['metric_mean'], 12, 4)}"
            f"{fmt(L.get('comb_score'))}"
            f"{fmt(L.get('score_p'))}"
            f"{fmt(capture_display)}"
            f"{fmt(L.get('breach_rate'))}"
            f"{fmt(L.get('mean_reward'), 10, 2)}"
            f"{s['status']}"
        )
    print()

    if n_contam:
        print(
            f"WARNING: {n_contam} report entries across all trials have "
            "full_capture_rate + breach_rate > 1.0. The env terminates on "
            "either event so these are mutually exclusive — the MetricsCallback "
            "is double-counting captures (see metrics.py:on_episode_step)."
        )
    if n_nan:
        print(f"WARNING: {n_nan} NaN values found across critical metric fields.")

    # Surface idle trials separately — they're the early signal that
    # the server died or a trial wedged on a gRPC call. The leaderboard
    # already shows status, but call them out by name for ops triage.
    idle_trials = [s["trial_id"] for s in summaries if s["status"].startswith("IDLE")]
    if idle_trials:
        print(f"IDLE trials: {', '.join(idle_trials)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
