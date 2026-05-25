from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyADRL.pool_metrics.models.evaluation_result import EvaluationPoolMetrics, EvaluationResult, combine

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = Path("average_table.typst")


def _short_name(pool: EvaluationPoolMetrics) -> str:
    import re

    cfg_match = re.search(r"(\d+)", pool.pursuer_config)
    trg_match = re.search(r"(\d+)", pool.pursuer_training)
    if cfg_match and trg_match:
        return f"c{int(cfg_match.group(1))}_t{int(trg_match.group(1))}"
    return f"{pool.pursuer_config}_{pool.pursuer_training}".strip("_")


def _map_row(pool: EvaluationPoolMetrics) -> dict:
    result: EvaluationResult = combine(pool.metrics)

    return {
        "name": _short_name(pool),
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


def generate_table(pools: list[EvaluationPoolMetrics], output: Path = DEFAULT_OUTPUT) -> None:
    rows = [_map_row(p) for p in pools]
    typst = build_typst_table(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(typst, encoding="utf-8")
    print(f"Wrote typst table to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Typst table from EvaluationPoolMetrics"
    )
    parser.add_argument(
        "--output",
        help=f"Output typst file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else DEFAULT_OUTPUT

    # Obtain pools from your source here.
    pools: list[EvaluationPoolMetrics] = []  # TODO: populate from your source

    generate_table(pools, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
