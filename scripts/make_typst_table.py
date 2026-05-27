from __future__ import annotations
from logging import Logger, log
import logging
import re
from dependency_injector.wiring import Provide, inject

import argparse
import sys
from pathlib import Path

from PyADRL.pool_metrics.models.evaluation_result import EvaluationPoolMetrics, EvaluationResult, combine
from PyADRL.pool_metrics.services.metrics_finder import MetricsFinder
from scripts.default_container import DefaultContainer

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = Path("average_table.typst")


def _map_row(name: str, result: EvaluationResult) -> dict:
    return {
        "name": name,
        "CR@1": result.capture_rate_at_k["1"],
        "ACS@1": result.mean_capture_step,
        "BR": result.breach_rate,
        "PTR": result.mean_pursuer_entered_target_rate,
        "ColR_P": result.mean_pursuer_drone_collision_rate,
        "OCR_P": result.mean_pursuer_obstacle_collision_rate,
        "BVR_P": result.mean_pursuer_out_of_bounds_rate,
        "OCR_E": result.mean_evader_obstacle_collision_rate,
        "BVR_E": result.mean_evader_out_of_bounds_rate,
    }


def _fmt(val) -> str:
    if val is None:
        return "—"
    try:
        if isinstance(val, (int, float)):
            if abs(val - int(val)) < 1e-6:
                return str(int(val))
            return f"{val:.3f}"
    except Exception:
        pass
    return str(val)


def build_typst_table(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: 10,")
    lines.append("    stroke: none,")
    lines.append("    align: auto,")
    lines.append("    inset: 3pt,")
    lines.append("    // highlight rule (example)")
    lines.append(
        "    fill: (x, y) => if (y == 6 and x == 1) or (y == 5 and x == 2) or (y == 6 and x == 3) or (y == 10 and x == 5) or (y == 9 and x == 6) or (y == 9 and x == 6) or (y == 4 and x == 9) or (y == 9 and x == 10) or (y == 8 and x == 10) { aau-blue.lighten(70%) },"
    )
    lines.append("    table.header(")

    lines.append("      table.cell(rowspan: 1)[],")
    lines.append("      table.vline(start: 0),")
    lines.append("      table.cell(colspan: 4, align: center)[*Performance*],")
    lines.append("      table.vline(start: 0),")
    lines.append("      table.cell(colspan: 5, align: center)[*Safety*],")

    header_parts = [
        "[Model]",
        '[$"CR@1"arrow.b$]',
        '[$"ACS@1"arrow.b$]',
        '[$"BR"arrow.b$]',
        '[$"PTR"arrow.b$]',
        '[$"ColR"_cal(P)arrow.b$]',
        '[$"OCR"_cal(P) arrow.b$]',
        '[$"BVR"_cal(P)arrow.b$]',
        '[$"OCR"_cal(E) arrow.b$]',
        '[$"BVR"_cal(E) arrow.b$]',
    ]
    for part in header_parts:
        lines.append(f"      {part},")

    lines.append("    ),")
    lines.append("    table.hline(start: 0),")

    for r in rows:
        lines.append(f"    meta[{r['name']}],")
        values = [
            r.get("CR@1"),
            r.get("ACS@1"),
            r.get("BR"),
            r.get("PTR"),
            r.get("ColR_P"),
            r.get("OCR_P"),
            r.get("BVR_P"),
            r.get("OCR_E"),
            r.get("BVR_E"),
        ]
        formatted = ", ".join(f"[{_fmt(v)}]" for v in values)
        lines.append(f"    {formatted},")
        lines.append('    table.hline(start: 1, stroke: (dash: "dotted")),')

    lines.append("  ),")
    lines.append("  caption: [  ]")
    lines.append(")")
    return "\n".join(lines)


def generate_table(pools: dict[str, EvaluationResult], output: Path = DEFAULT_OUTPUT) -> None:
    rows = [_map_row(k, p) for k, p in pools.items()]
    typst = build_typst_table(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(typst, encoding="utf-8")
    print(f"Wrote typst table to {output}")

def get_number(text: str):
    return int(re.search("[0-9]+", text).group())

def get_type_conf(name: str, experiment: str):
    if re.match(r"s[1-3]$", name):
        return "s"
    if re.match(r"ss[1-3]$", name):
        return "ss"
    if re.match(r"sa[1-3]$", name):
        return "sa"
    if re.match(r"a[1-3]$", name):
        return "a"
    if re.match(r"l[1-3]$", name):
        return "l"
    if re.match(r"c[1-3]$", name):
        return experiment[0]
    raise Exception("did not recognize type")



@inject
def main(
    logger: Logger = Provide[DefaultContainer.logger],
    metrics_finder: MetricsFinder = Provide[DefaultContainer.metrics_finder],
) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Typst table from EvaluationPoolMetrics"
    )
    parser.add_argument(
        "--output",
        "-o",
        help=f"Output typst file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else DEFAULT_OUTPUT

    # Obtain pools from your source here.
    pools: list[EvaluationPoolMetrics] = [m for m in metrics_finder.scan_for_metrics() if re.match(r'(long-v-all)$', m.experiment_name)]
    
    results: dict[str, list[EvaluationResult]] = {}
    for pool in pools:
        key = f"{get_type_conf(pool.pursuer_config, pool.experiment_name)}-c{get_number(pool.pursuer_config)}-t{get_number(pool.pursuer_training)}-{get_type_conf(pool.evader_config, pool.experiment_name)}"

        if key not in results:
            results[key] = []

        metrics = results[key]
        metrics.extend(pool.metrics)


    combined_metrics: dict[str, EvaluationResult] = {}
    for key, value in results.items():
        logger.info("%s: count(%s)", key, len(value))
        combined_metrics[key] = combine(value)

    generate_table(combined_metrics, output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    container = DefaultContainer()
    container.wire(modules=[__name__])

    raise SystemExit(main())
