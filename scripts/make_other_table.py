from __future__ import annotations
from logging import Logger, log, raiseExceptions
import numpy as np
import logging
import re
from dependency_injector.wiring import Provide, inject

import argparse
import sys
from pathlib import Path

from PyADRL.pool_metrics.models.evaluation_result import EvaluationPoolMetrics, EvaluationResult, combine
from PyADRL.pool_metrics.services.metrics_finder import MetricsFinder
from scripts.default_container import DefaultContainer


class TableRow:
    def __init__(
        self,
        name: str,
        cr1: float,
        acs1: float,
        br: float,
        ptr: float,
        colr_p: float,
        ocr_p: float,
        bvr_p: float,
        ocr_e: float,
        bvr_e: float,
        cr1_std: float,
        acs1_std: float,
        br_std: float,
        ptr_std: float,
        colr_p_std: float,
        ocr_p_std: float,
        bvr_p_std: float,
        ocr_e_std: float,
        bvr_e_std: float
    ) -> None:
        self.name = name

        self.cr1 = cr1
        self.cr1_std = cr1_std

        self.acs1 = acs1
        self.acs1_std = acs1_std

        self.br = br
        self.br_std = br_std

        self.ptr = ptr
        self.ptr_std = ptr_std

        self.colr_p = colr_p
        self.colr_p_std = colr_p_std

        self.ocr_p = ocr_p
        self.ocr_p_std = ocr_p_std

        self.bvr_p = bvr_p
        self.bvr_p_std = bvr_p_std

        self.ocr_e = ocr_e
        self.ocr_e_std = ocr_e_std

        self.bvr_e = bvr_e
        self.bvr_e_std = bvr_e_std


class TableBuilder:
    _HEAD = """
#figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
        align: (left, center, center, center, center, center, center, center, center, center),
        stroke: 0.4pt,
        inset: (x: 6pt, y: 4pt),

        // Header row
        table.header(
          [*Setting*],
          [CR\\@1 $arrow.t$],
          [ACS\\@1 $arrow.b$],
          [BR $arrow.b$],
          [PTR $arrow.b$],
          [ColRp $arrow.b$],
          [OCRp $arrow.b$],
          [BVRp $arrow.b$],
          [OCRe $arrow.b$],
          [BVRe $arrow.b$],
        ),
    """
    _TAIL = """
          ),
          caption: none,
        )
    """

    def build_table(self, rows: list[TableRow]) -> str:
        ss = [self._HEAD] + [self.row_format(r) for r in rows] + [self._TAIL]
        return "\n".join(ss)

    def row_format(self, row: TableRow):
        return f"""
            // Row 1: alt p vs alt e
            [{row.name}],
            [${row.cr1:.2f} plus.minus {row.cr1_std:.2f}$],
            [${row.acs1:.2f} plus.minus {row.acs1_std:.2f}$],
            [${row.br:.2f} plus.minus {row.br_std:.2f}$],
            [${row.ptr:.2f} plus.minus {row.ptr_std:.2f}$],
            [${row.colr_p:.2f} plus.minus {row.colr_p_std:.2f}$],
            [${row.ocr_p:.2f} plus.minus {row.ocr_p_std:.2f}$],
            [${row.bvr_p:.2f} plus.minus {row.bvr_p_std:.2f}$],
            [${row.ocr_e:.2f} plus.minus {row.ocr_e_std:.2f}$],
            [${row.bvr_e:.2f} plus.minus {row.bvr_e_std:.2f}$],
"""

def row_from_result(name: str, results: list[EvaluationResult]) -> TableRow:
    return TableRow(
        name,

        cr1=float(np.mean([result.capture_rate_at_k["1"] for result in results])),
        acs1=float(np.mean([result.mean_capture_step for result in results])),
        br=float(np.mean([result.breach_rate for result in results])),
        ptr=float(np.mean([result.mean_pursuer_entered_target_rate for result in results])),
        colr_p=float(np.mean([result.mean_pursuer_drone_collision_rate for result in results])),
        ocr_p=float(np.mean([result.mean_pursuer_obstacle_collision_rate for result in results])),
        bvr_p=float(np.mean([result.mean_pursuer_out_of_bounds_rate for result in results])),
        ocr_e=float(np.mean([result.mean_evader_obstacle_collision_rate for result in results])),
        bvr_e=float(np.mean([result.mean_evader_out_of_bounds_rate for result in results])),

        cr1_std=float(np.std([result.capture_rate_at_k["1"] for result in results])),
        acs1_std=float(np.std([result.mean_capture_step for result in results])),
        br_std=float(np.std([result.breach_rate for result in results])),
        ptr_std=float(np.std([result.mean_pursuer_entered_target_rate for result in results])),
        colr_p_std=float(np.std([result.mean_pursuer_drone_collision_rate for result in results])),
        ocr_p_std=float(np.std([result.mean_pursuer_obstacle_collision_rate for result in results])),
        bvr_p_std=float(np.std([result.mean_pursuer_out_of_bounds_rate for result in results])),
        ocr_e_std=float(np.std([result.mean_evader_obstacle_collision_rate for result in results])),
        bvr_e_std=float(np.std([result.mean_evader_out_of_bounds_rate for result in results])),
    )

@inject
def main(
    logger: Logger = Provide[DefaultContainer.logger],
    metrics_finder: MetricsFinder = Provide[DefaultContainer.metrics_finder],
) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Typst table from EvaluationPoolMetrics"
    )
    args = parser.parse_args()

    # Obtain pools from your source here.
    pools: list[EvaluationPoolMetrics] = [m for m in metrics_finder.scan_for_metrics() if re.match(r"(a-v-sa|s-v-ss)$", m.experiment_name)]

    config_pools: dict[tuple[str, str, str], list[EvaluationPoolMetrics]] = {}
    for p in pools:
        amatchp = re.match("a[1-3]$", p.pursuer_config)
        samatchp = re.match("sa[1-3]$", p.pursuer_config)
        smatchp = re.match("s[1-3]$", p.pursuer_config)
        ssmatchp = re.match("ss[1-3]$", p.pursuer_config)

        ptype = "ERR"
        if amatchp:
            ptype = "a"
        elif samatchp:
            ptype = "sa"
        elif smatchp:
            ptype = "s"
        elif ssmatchp:
            ptype = "ss"
        else:
            raise Exception("f")
        
        amatche = re.match("a[1-3]$", p.evader_config)
        samatche = re.match("sa[1-3]$", p.evader_config)
        smatche = re.match("s[1-3]$", p.evader_config)
        ssmatche = re.match("ss[1-3]$", p.evader_config)

        etype = "ERR"
        if amatche:
            etype = "a"
        elif samatche:
            etype = "sa"
        elif smatche:
            etype = "s"
        elif ssmatche:
            etype = "ss"
        else:
            raise Exception("f")
        
        key = (ptype, p.pursuer_config, etype)

        if key not in config_pools:
            config_pools[key] = []

        config_pools[key].append(p)

    avged: dict[tuple[str, str], list[EvaluationResult]] = {}
    for (ptype, _, etype), pools in config_pools.items():
        metrics = combine([m for p in pools for m in p.metrics])

        key = (ptype, etype)

        if key not in avged:
            avged[key] = []
        avged[key].append(metrics)

    
    rows = [row_from_result(f"{conf}-{evad}", rs) for (conf, evad), rs in avged.items()]

    Path("./some_table.typst").write_text(TableBuilder().build_table(rows))

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    container = DefaultContainer()
    container.wire(modules=[__name__])

    raise SystemExit(main())
