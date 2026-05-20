"""Sanity-check post-tune trained models for completeness and metric leakage.

Usage:
    python check_models.py experiments/experiment_4
    python check_models.py experiments/experiment_4 250       # expected iters
    python check_models.py experiments/experiment_4 250 1     # expected iters, n_evaders

Handles trainings in any state:
  - Not started (no eval file yet)        -> reported, not errored
  - In progress (some iters done)         -> progress shown, contamination
                                              still checked on data so far
  - Crashed mid-training                  -> errored, with partial data
  - Completed                             -> full sanity check

Key checks on completed trainings:
  1. Expected files exist (final_model, evaluation_metrics.json)
  2. Training reached expected iteration count
  3. Metric values in valid ranges
  4. Invariant: capture_rate_at_k[n_evaders] + breach_rate <= 1.0
     (env terminates on either event - mutually exclusive)
  5. No NaN in critical fields (incl. comb_score / score_p / capture_score
     when present — runs predating the composite-score impl just skip these)
  6. mean_rewards dict has both pursuer + evader sides
  7. No catastrophic regression in final eval vs mid-training
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TOL = 1e-3


class Issue:
    def __init__(self, severity: str, training_dir: Path, message: str):
        self.severity = severity  # ERROR, WARN, INFO
        self.training_dir = training_dir
        self.message = message

    def label(self) -> str:
        if self.training_dir is None:
            return "?"
        return f"{self.training_dir.parent.name}/{self.training_dir.name}"

    def __str__(self):
        return f"[{self.severity:5s}] {self.label()}: {self.message}"


def get_full_capture_rate(entry: dict, n_evaders: int) -> float | None:
    """Extract full capture rate from capture_rate_at_k dict."""
    rates = entry.get("capture_rate_at_k")
    if not isinstance(rates, dict):
        return None
    # Keys may be ints or strings depending on JSON round-trip
    value = rates.get(n_evaders, rates.get(str(n_evaders)))
    return float(value) if value is not None else None


def get_agent_rewards(entry: dict) -> tuple[float | None, float | None]:
    """Return (pursuer_reward_sum, evader_reward_sum) from mean_rewards dict.

    Agent names follow 'pursuer_0', 'pursuer_1', 'evader_0', etc. We sum
    per-side because the per-team total is what's meaningful for comparison.
    """
    mean_rewards = entry.get("mean_rewards")
    if not isinstance(mean_rewards, dict):
        return None, None
    pursuer = 0.0
    evader = 0.0
    p_count = 0
    e_count = 0
    for agent_name, reward in mean_rewards.items():
        if reward is None:
            continue
        if "pursuer" in agent_name.lower():
            pursuer += reward
            p_count += 1
        elif "evader" in agent_name.lower():
            evader += reward
            e_count += 1
    return (pursuer if p_count else None, evader if e_count else None)


def classify_training_state(training_dir: Path, expected_iters: int) -> str:
    """Return one of: 'not_started', 'in_progress', 'crashed', 'complete'."""
    eval_file = training_dir / "evaluation_metrics.json"
    final_model = training_dir / "final_model"

    if not eval_file.exists():
        return "not_started"

    try:
        with open(eval_file) as f:
            data = json.load(f)
    except Exception:
        return "crashed"

    if not isinstance(data, list) or len(data) == 0:
        return "not_started"

    n_evals = len(data)
    if final_model.exists() and n_evals >= expected_iters * 0.9:
        return "complete"
    if final_model.exists():
        return "crashed"
    return "in_progress"


def check_metrics_completed(
    training_dir: Path, n_evaders: int, expected_iters: int, issues: list[Issue]
):
    """Full sanity check for a completed training."""
    eval_file = training_dir / "evaluation_metrics.json"
    try:
        with open(eval_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(Issue("ERROR", training_dir, f"Malformed JSON: {e}"))
        return

    if not isinstance(data, list) or not data:
        issues.append(Issue("ERROR", training_dir, "evaluation_metrics.json empty"))
        return

    # ---- Aggregate problems across entries; report once per type ----
    missing_keys: dict[str, int] = defaultdict(int)
    out_of_range: dict[str, int] = defaultdict(int)
    invariant_violations = 0
    invariant_examples: list[tuple[int, float, float]] = []
    nan_count = 0

    last = data[-1]

    for i, entry in enumerate(data):
        for key in ("breach_rate", "mean_episode_length", "mean_capture_step"):
            if key not in entry:
                missing_keys[key] += 1

        fc = get_full_capture_rate(entry, n_evaders)
        br = entry.get("breach_rate")
        pursuer_r, evader_r = get_agent_rewards(entry)

        # Composite scores: present only when evaluation_metrics.json was
        # written after the score_p/comb_score impl landed. Treat as
        # optional — older runs just won't be checked on these fields.
        comb = entry.get("comb_score")
        score_p = entry.get("score_p")
        cap_score = entry.get("capture_score")
        wacs = entry.get("weighted_acs")

        for v in (fc, br, pursuer_r, evader_r, comb, score_p, cap_score, wacs):
            if isinstance(v, float) and v != v:  # NaN
                nan_count += 1

        if fc is not None and not (-TOL <= fc <= 1.0 + TOL):
            out_of_range["full_capture_rate"] += 1
        if br is not None and not (-TOL <= br <= 1.0 + TOL):
            out_of_range["breach_rate"] += 1
        # capture_score is bounded in [0, 1] — same domain as a rate.
        if cap_score is not None and not (-TOL <= cap_score <= 1.0 + TOL):
            out_of_range["capture_score"] += 1
        # weighted_acs is a time in [0, Tmax*(|E|+1)/2]; we don't know
        # Tmax here so just sanity-check non-negativity.
        if wacs is not None and wacs < -TOL:
            out_of_range["weighted_acs (negative)"] += 1
        # score_p / comb_score have no natural upper bound but should never
        # exceed +1 in practice (capture_score caps the positive term at 1).
        # Catch obviously-broken values (very large magnitudes); the bound
        # is loose because beta*ACS/Tmax can in theory exceed 1.
        for name, val in (("score_p", score_p), ("comb_score", comb)):
            if val is not None and not (-10.0 <= val <= 10.0):
                out_of_range[name] += 1

        if fc is not None and br is not None and fc + br > 1.0 + TOL:
            invariant_violations += 1
            if len(invariant_examples) < 3:
                invariant_examples.append((i, fc, br))

        # capture_rate_at_k buckets should sum to ~1.0 (every episode is in
        # exactly one bucket k = number of evaders captured). If they don't,
        # the callback isn't classifying episodes correctly.
        crk = entry.get("capture_rate_at_k")
        if isinstance(crk, dict) and crk:
            total = sum(v for v in crk.values() if isinstance(v, (int, float)))
            if abs(total - 1.0) > TOL:
                out_of_range[f"capture_rate_at_k sum ({total:.3f}, expected ~1.0)"] += 1

    for key, count in missing_keys.items():
        issues.append(
            Issue(
                "ERROR", training_dir, f"Missing '{key}' in {count}/{len(data)} entries"
            )
        )
    for key, count in out_of_range.items():
        issues.append(
            Issue(
                "ERROR",
                training_dir,
                f"'{key}' out of range in {count}/{len(data)} entries",
            )
        )
    if invariant_violations > 0:
        examples = ", ".join(
            f"entry {i}: {fc:.2f}+{br:.2f}={fc + br:.2f}"
            for i, fc, br in invariant_examples
        )
        issues.append(
            Issue(
                "ERROR",
                training_dir,
                f"capture+breach > 1.0 in {invariant_violations}/{len(data)} entries "
                f"(callback contamination?) [{examples}]",
            )
        )
    if nan_count > 0:
        issues.append(Issue("WARN", training_dir, f"Found {nan_count} NaN values"))

    # Convergence sanity
    # Final eval should not be dramatically worse than mid-training average.
    # Use a relative threshold (15% drop) instead of absolute — rewards can
    # be on very different scales depending on the env's reward shaping, so
    # an absolute "5 reward units" threshold fires false positives on
    # higher-magnitude reward functions.
    if len(data) >= 20:
        final_p, _ = get_agent_rewards(last)
        mid_rewards = []
        for e in data[len(data) // 2 : -1]:
            p, _ = get_agent_rewards(e)
            if p is not None:
                mid_rewards.append(p)
        if final_p is not None and mid_rewards:
            mid_mean = sum(mid_rewards) / len(mid_rewards)
            # Use a relative drop threshold so we don't false-positive on
            # high-magnitude rewards. Skip the check entirely if mid_mean
            # is near zero (relative threshold becomes meaningless).
            if abs(mid_mean) > 10:
                drop_pct = (mid_mean - final_p) / abs(mid_mean) * 100
                if drop_pct > 15.0:
                    issues.append(
                        Issue(
                            "WARN",
                            training_dir,
                            f"Final pursuer_reward ({final_p:.2f}) is {drop_pct:.0f}% "
                            f"below mid-training mean ({mid_mean:.2f}). Possible collapse.",
                        )
                    )

    if len(data) < expected_iters * 0.9:
        issues.append(
            Issue(
                "WARN",
                training_dir,
                f"Only {len(data)} eval entries (expected ~{expected_iters + 1})",
            )
        )

    # Final results summary
    fc = get_full_capture_rate(last, n_evaders)
    br = last.get("breach_rate")
    p_r, e_r = get_agent_rewards(last)
    el = last.get("mean_episode_length")
    cs = last.get("mean_capture_step")
    e_oob = last.get("mean_evader_out_of_bounds_rate")
    comb = last.get("comb_score")
    score_p = last.get("score_p")
    cap_score = last.get("capture_score")
    parts = []
    # Lead with comb_score / score_p / capture_score (the selection-metric
    # family) when present — these are the numbers the tuner ranked on.
    if comb is not None:
        parts.append(f"comb={comb:.3f}")
    if score_p is not None:
        parts.append(f"score_p={score_p:.3f}")
    if cap_score is not None:
        parts.append(f"cap_score={cap_score:.3f}")
    if fc is not None:
        parts.append(f"capture={fc:.3f}")
    if br is not None:
        parts.append(f"breach={br:.3f}")
    if fc is not None and br is not None:
        parts.append(f"success={fc - br:.3f}")
    if p_r is not None:
        parts.append(f"pursuer_r={p_r:.1f}")
    if e_r is not None:
        parts.append(f"evader_r={e_r:.1f}")
    if el is not None:
        parts.append(f"ep_len={el:.1f}")
    if cs is not None:
        parts.append(f"cap_step={cs:.1f}")
    if e_oob is not None:
        parts.append(f"e_oob={e_oob:.2f}")
    if parts:
        issues.append(Issue("INFO", training_dir, "  ".join(parts)))


def check_metrics_in_progress(
    training_dir: Path, n_evaders: int, expected_iters: int, issues: list[Issue]
):
    """Lightweight check for trainings still running."""
    eval_file = training_dir / "evaluation_metrics.json"
    try:
        with open(eval_file) as f:
            data = json.load(f)
    except Exception:
        return

    if not isinstance(data, list) or not data:
        return

    n_done = len(data)
    pct = n_done / expected_iters * 100 if expected_iters > 0 else 0
    last = data[-1]
    fc = get_full_capture_rate(last, n_evaders)
    br = last.get("breach_rate")
    p_r, _ = get_agent_rewards(last)
    comb = last.get("comb_score")
    parts = [f"iter ~{n_done}/{expected_iters} ({pct:.0f}%)"]
    if comb is not None:
        parts.append(f"comb={comb:.3f}")
    if fc is not None:
        parts.append(f"capture={fc:.3f}")
    if br is not None:
        parts.append(f"breach={br:.3f}")
    if p_r is not None:
        parts.append(f"pursuer_r={p_r:.1f}")
    issues.append(Issue("INFO", training_dir, "IN PROGRESS — " + "  ".join(parts)))

    contaminated = 0
    for entry in data:
        f_ = get_full_capture_rate(entry, n_evaders)
        b_ = entry.get("breach_rate")
        if f_ is not None and b_ is not None and f_ + b_ > 1.0 + TOL:
            contaminated += 1
    if contaminated > 0:
        issues.append(
            Issue(
                "ERROR",
                training_dir,
                f"capture+breach > 1.0 in {contaminated}/{len(data)} entries "
                "(callback contamination - consider killing and restarting)",
            )
        )


def check_experiment(
    experiment_dir: Path, expected_iters: int = 250, n_evaders: int = 1
):
    if not experiment_dir.exists():
        print(f"ERROR: {experiment_dir} does not exist")
        return 1

    issues: list[Issue] = []
    state_counts: dict[str, int] = defaultdict(int)

    config_dirs = sorted(experiment_dir.glob("config_*"))
    if not config_dirs:
        print(f"ERROR: No config_* directories under {experiment_dir}")
        return 1

    n_total = 0
    for config_dir in config_dirs:
        for training_dir in sorted(config_dir.glob("training_*")):
            n_total += 1
            state = classify_training_state(training_dir, expected_iters)
            state_counts[state] += 1

            if state == "not_started":
                issues.append(
                    Issue(
                        "INFO", training_dir, "NOT STARTED (no evaluation_metrics.json)"
                    )
                )
            elif state == "in_progress":
                check_metrics_in_progress(
                    training_dir, n_evaders, expected_iters, issues
                )
            elif state == "crashed":
                issues.append(
                    Issue(
                        "ERROR",
                        training_dir,
                        "CRASHED (eval file present, final_model missing)",
                    )
                )
                check_metrics_in_progress(
                    training_dir, n_evaders, expected_iters, issues
                )
            elif state == "complete":
                check_metrics_completed(training_dir, n_evaders, expected_iters, issues)

    print(f"\n=== {experiment_dir} ===")
    print(
        f"Trainings: {n_total} total, "
        f"{state_counts.get('complete', 0)} complete, "
        f"{state_counts.get('in_progress', 0)} in progress, "
        f"{state_counts.get('not_started', 0)} not started, "
        f"{state_counts.get('crashed', 0)} crashed"
    )
    print()

    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARN"]
    infos = [i for i in issues if i.severity == "INFO"]

    if errors:
        print(f"--- {len(errors)} ERROR(S) ---")
        for issue in errors:
            print(f"  {issue}")
        print()

    if warnings:
        print(f"--- {len(warnings)} WARNING(S) ---")
        for issue in warnings:
            print(f"  {issue}")
        print()

    if infos:
        print(f"--- RESULTS ({len(infos)}) ---")
        for issue in sorted(infos, key=lambda x: x.label()):
            print(f"  {issue}")
        print()

    if not errors:
        if state_counts.get("complete", 0) == n_total:
            print("All trainings complete and healthy. ✓\n")
        else:
            print("No errors so far. ✓\n")

    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    experiment_dir = Path(sys.argv[1])
    expected_iters = int(sys.argv[2]) if len(sys.argv) >= 3 else 250
    n_evaders = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
    sys.exit(check_experiment(experiment_dir, expected_iters, n_evaders))
