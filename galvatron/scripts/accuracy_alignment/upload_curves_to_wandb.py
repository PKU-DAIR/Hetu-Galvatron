#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import wandb


LOSS_LINE_PATTERN = re.compile(
    r"\|\s*Iteration:\s*(\d+)\s*\|.*?\|\s*Loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\|"
)
LOSS_LINE_PATTERN_BRACKET = re.compile(
    r"\(Iteration\s+(\d+)\):\s*Loss\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _parse_loss_lines(log_path: Path):
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOSS_LINE_PATTERN.search(line) or LOSS_LINE_PATTERN_BRACKET.search(line)
        if not match:
            continue
        rows.append({"iteration": int(match.group(1)), "loss": float(match.group(2))})
    return rows


def _write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upload_curve(project: str, run_name: str, mode: str, curve_csv: Path, entity: str = ""):
    rows = _read_csv(curve_csv)
    metric_name = "loss"
    run = wandb.init(project=project, name=run_name, entity=entity or None)
    run.summary["mode"] = mode
    run.summary["curve_file"] = str(curve_csv)
    run.summary["metric_name"] = metric_name
    for row in rows:
        step = int(row["iteration"])
        loss = float(row["loss"])
        run.log(
            {
                metric_name: loss,
                f"{mode}_loss": loss,
                "iteration": step,
            },
            step=step,
        )
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Extract loss curve from log and upload to wandb.")
    parser.add_argument("--mode", choices=["baseline", "test"], required=True)
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--run-name", required=True, help="wandb run name")
    parser.add_argument("--entity", default="", help="wandb entity/team")
    parser.add_argument("--curve-csv", required=True, help="curve csv path")
    parser.add_argument("--log", required=True, help="training log path")
    args = parser.parse_args()

    rows = _parse_loss_lines(Path(args.log))
    if not rows:
        raise RuntimeError(f"No loss lines found in log: {args.log}")
    _write_csv(rows, Path(args.curve_csv))
    print(f"Extracted {len(rows)} points to {args.curve_csv}")

    upload_curve(
        project=args.project,
        run_name=args.run_name,
        mode=args.mode,
        curve_csv=Path(args.curve_csv),
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
