#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so imports like `import evaluation` work
SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import evaluation


DEFAULT_ROOT = Path("experiment_2")
DEFAULT_OUTPUT = DEFAULT_ROOT / "average_table.typst"


# Mapping row values: remove MAP and shorten naming to c{n}_t{m}
def _short_name(summary: dict[str, Any]) -> str:
    cfg = summary.get("config", "")
    trg = summary.get("training", "")
    cfg_match = re.search(r"(\d+)", cfg)
    trg_match = re.search(r"(\d+)", trg)
    if cfg_match and trg_match:
        return f"c{int(cfg_match.group(1))}_t{int(trg_match.group(1))}"
    # fallback to compact original
    return f"{cfg}_{trg}".strip("_")


def _map_row(summary: dict[str, Any]) -> dict[str, Any]:
    # Capture rate at k: prefer key "1" for CR@1, fall back to "0" then capture_score
    cr = summary.get("capture_rate_at_k")
    capture_rate = None
    if isinstance(cr, dict):
        if "1" in cr:
            capture_rate = cr["1"]
        elif "0" in cr:
            capture_rate = cr["0"]
        else:
            vals = list(cr.values())
            if vals:
                # prefer second element if available
                capture_rate = vals[1] if len(vals) > 1 else vals[0]
    if capture_rate is None:
        capture_rate = summary.get("capture_score")

    # ACS@1: prefer mean_capture_step_at_k[1], fall back to [0], then mean_capture_step
    acs = None
    acs_list = summary.get("mean_capture_step_at_k")
    if isinstance(acs_list, list) and acs_list:
        if len(acs_list) > 1 and acs_list[1] is not None:
            acs = acs_list[1]
        else:
            acs = acs_list[0]
    if acs is None:
        acs = summary.get("mean_capture_step")

    # breach rate
    breach = summary.get("breach_rate")

    # pursuer in target
    ptr = summary.get("mean_pursuer_entered_target_rate")

    def pick(key: str) -> Any:
        return summary.get(key)

    return {
        "name": _short_name(summary),
        "CR@1": capture_rate,
        "ACS@1": acs,
        "BR": breach,
        "PTR": ptr,
        "ColR_P": pick("mean_pursuer_drone_collision_rate"),
        "OCR_P": pick("mean_pursuer_obstacle_collision_rate"),
        "BVR_P": pick("mean_pursuer_out_of_bounds_rate"),
        "ColR_E": pick("mean_evader_drone_collision_rate"),
        "OCR_E": pick("mean_evader_obstacle_collision_rate"),
        "BVR_E": pick("mean_evader_out_of_bounds_rate"),
    }


def _fmt(val: Any) -> str:
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


def build_typst_table(rows: list[dict[str, Any]]) -> str:
    # 10 columns: Model + 4 Performance + 5 Safety
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

    # Top grouping row
    lines.append("      table.cell(rowspan: 1)[],")
    lines.append("      table.vline(start: 0),")
    lines.append("      table.cell(colspan: 4, align: center)[*Performance*],")
    lines.append("      table.vline(start: 0),")
    lines.append("      table.cell(colspan: 5, align: center)[*Safety*],")

    # Second header row (ColR(E) removed)
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

    # Rows: model name, then the 9 metrics in the same order as header_parts[1:]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Typst table from per-training average JSON files"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help="Root directory (default: experiment_2)",
    )
    parser.add_argument(
        "--output", help="Output typst file (default: experiment_2/average_table.typst)"
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    output = Path(args.output) if args.output else DEFAULT_OUTPUT

    summaries = evaluation.build_average_summaries(root)
    rows = [_map_row(s) for s in summaries]

    typst = build_typst_table(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(typst, encoding="utf-8")
    print(f"Wrote typst table to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
