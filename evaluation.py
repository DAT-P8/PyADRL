from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("experiment_2")
DEFAULT_OUTPUT_NAME = "average_evaluation_metrics.json"
METRIC_FILE_NAME = "evaluation_metrics.json"
SKIPPED_KEYS = {"timestamp"}


@dataclass
class MetricAccumulator:
    kind: str | None = None
    total: float = 0.0
    count: int = 0
    children: dict[Any, "MetricAccumulator"] = field(default_factory=dict)

    def _child(self, key: Any) -> "MetricAccumulator":
        child = self.children.get(key)
        if child is None:
            child = MetricAccumulator()
            self.children[key] = child
        return child

    def add(self, value: Any) -> None:
        if _is_number(value):
            if self.kind is None:
                self.kind = "number"
            if self.kind != "number":
                return
            self.total += float(value)
            self.count += 1
            return

        if isinstance(value, Mapping):
            if self.kind is None:
                self.kind = "dict"
            if self.kind != "dict":
                return
            for key, child_value in value.items():
                if key in SKIPPED_KEYS:
                    continue
                self._child(key).add(child_value)
            return

        if isinstance(value, list):
            if self.kind is None:
                self.kind = "list"
            if self.kind != "list":
                return
            for index, child_value in enumerate(value):
                self._child(index).add(child_value)

    def finalize(self) -> Any:
        if self.kind == "number":
            return self.total / self.count if self.count else None

        if self.kind == "dict":
            dict_result: dict[Any, Any] = {}
            for key, child in self.children.items():
                value = child.finalize()
                if value is not None:
                    dict_result[key] = value
            return dict_result

        if self.kind == "list":
            if not self.children:
                return []
            max_index = max(int(index) for index in self.children)
            list_result: list[Any] = []
            for index in range(max_index + 1):
                child = self.children.get(index)
                list_result.append(child.finalize() if child is not None else None)
            while list_result and list_result[-1] is None:
                list_result.pop()
            return list_result

        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _natural_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"_(\d+)$", path.name)
    return (path.name.split("_")[0], int(match.group(1)) if match else -1, path.name)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_metric_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return [payload]

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def iter_metric_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(METRIC_FILE_NAME) if path.is_file())


def summarize_training(training_root: Path) -> dict[str, Any]:
    metric_files = iter_metric_files(training_root)
    accumulator = MetricAccumulator(kind="dict")
    skipped_files: list[str] = []
    records_used = 0

    for path in metric_files:
        try:
            records = load_metric_records(path)
        except Exception:
            skipped_files.append(_relative_path(training_root, path))
            continue

        if not records:
            skipped_files.append(_relative_path(training_root, path))
            continue

        for record in records:
            records_used += 1
            for key, value in record.items():
                if key in SKIPPED_KEYS:
                    continue
                accumulator._child(key).add(value)

    summary: dict[str, Any] = {
        "config": training_root.parent.name,
        "training": training_root.name,
        "training_path": str(training_root),
        "files_found": len(metric_files),
        "records_used": records_used,
        "skipped_files": skipped_files,
    }

    averaged_metrics = accumulator.finalize() or {}
    if isinstance(averaged_metrics, dict):
        summary.update(averaged_metrics)
    return summary


def iter_training_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []

    configs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("config_")
        ),
        key=_natural_sort_key,
    )
    training_dirs: list[Path] = []
    for config_dir in configs:
        trainings = sorted(
            (
                path
                for path in config_dir.iterdir()
                if path.is_dir() and path.name.startswith("training_")
            ),
            key=_natural_sort_key,
        )
        training_dirs.extend(trainings)
    return training_dirs


def build_average_summaries(root: Path) -> list[dict[str, Any]]:
    return [
        summarize_training(training_dir) for training_dir in iter_training_dirs(root)
    ]


def write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_training_summaries(root: Path, output_name: str) -> list[Path]:
    output_paths: list[Path] = []
    for training_summary in build_average_summaries(root):
        output_path = Path(training_summary["training_path"]) / output_name
        write_summary(output_path, training_summary)
        output_paths.append(output_path)
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average evaluation metrics for each training directory in an experiment tree."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help="Root directory to scan recursively (default: experiment_2).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Filename to write inside each training directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    output_paths = write_training_summaries(root, args.output_name)
    print(f"Saved {len(output_paths)} averaged evaluation files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
