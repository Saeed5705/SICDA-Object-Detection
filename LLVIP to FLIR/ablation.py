"""Fully executable SICDA ablation runner.

Runs seven cumulative experiments as fresh Python processes so model state,
optimizer state, CUDA memory, and disabled modules cannot leak between rows.

Default output:
    outputs/ablation/ablation_results.csv
    outputs/ablation/<experiment>/result.json
    outputs/ablation/<experiment>/checkpoints/best_model.pth
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from config import Config


ABLATION_EXPERIMENTS = [
    (
        "Baseline",
        dict(
            attention=False,
            illumination=False,
            ssl=False,
            domain=False,
            mmd=False,
            consistency=False,
        ),
    ),
    (
        "+ Cross-Modal Attention",
        dict(
            attention=True,
            illumination=False,
            ssl=False,
            domain=False,
            mmd=False,
            consistency=False,
        ),
    ),
    (
        "+ Illumination",
        dict(
            attention=True,
            illumination=True,
            ssl=False,
            domain=False,
            mmd=False,
            consistency=False,
        ),
    ),
    (
        "+ SSL",
        dict(
            attention=True,
            illumination=True,
            ssl=True,
            domain=False,
            mmd=False,
            consistency=False,
        ),
    ),
    (
        "+ GRL Domain Adaptation",
        dict(
            attention=True,
            illumination=True,
            ssl=True,
            domain=True,
            mmd=False,
            consistency=False,
        ),
    ),
    (
        "+ MMD",
        dict(
            attention=True,
            illumination=True,
            ssl=True,
            domain=True,
            mmd=True,
            consistency=False,
        ),
    ),
    (
        "Full SICDA",
        dict(
            attention=True,
            illumination=True,
            ssl=True,
            domain=True,
            mmd=True,
            consistency=True,
        ),
    ),
]


def slugify(name):
    s = name.lower().replace("+", "plus")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "experiment"


def bool_arg(value):
    return "1" if value else "0"


def parse_args():
    p = argparse.ArgumentParser(description="Run all SICDA ablations")
    p.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--workers", type=int, default=Config.NUM_WORKERS)
    p.add_argument(
        "--output-root", default=os.path.join(Config.OUTPUT_DIR, "ablation")
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a completed result.json instead of rerunning it.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional experiment names/slugs to run.",
    )
    return p.parse_args()


def selected_experiments(only):
    if not only:
        return ABLATION_EXPERIMENTS
    wanted = {x.lower() for x in only}
    selected = []
    for name, setting in ABLATION_EXPERIMENTS:
        if name.lower() in wanted or slugify(name).lower() in wanted:
            selected.append((name, setting))
    if not selected:
        valid = ", ".join(slugify(n) for n, _ in ABLATION_EXPERIMENTS)
        raise SystemExit(f"No experiment matched --only. Valid slugs: {valid}")
    return selected


def run_one(name, setting, args, output_root):
    slug = slugify(name)
    exp_dir = output_root / slug
    ckpt_dir = exp_dir / "checkpoints"
    result_json = exp_dir / "result.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and result_json.exists():
        print(f"\n[SKIP] {name}: using {result_json}")
        with open(result_json, "r", encoding="utf-8") as f:
            return json.load(f)

    command = [
        sys.executable,
        "train.py",
        "--experiment-name",
        name,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--attention",
        bool_arg(setting["attention"]),
        "--illumination",
        bool_arg(setting["illumination"]),
        "--ssl",
        bool_arg(setting["ssl"]),
        "--grl",
        bool_arg(setting["domain"]),
        "--mmd",
        bool_arg(setting["mmd"]),
        "--consistency",
        bool_arg(setting["consistency"]),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--output-dir",
        str(exp_dir),
        "--result-json",
        str(result_json),
    ]

    print("\n" + "=" * 78)
    print(f"Running ablation: {name}")
    print(f"Flags: {setting}")
    print(f"Output: {exp_dir}")
    print("=" * 78)

    completed = subprocess.run(command, cwd=Path(__file__).resolve().parent)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Ablation experiment '{name}' failed with exit code "
            f"{completed.returncode}. See console output above."
        )
    if not result_json.exists():
        raise RuntimeError(
            f"Training completed but result file was not created: {result_json}"
        )

    with open(result_json, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics, key, fallback=None):
    if key in metrics:
        return metrics[key]
    if fallback is not None and fallback in metrics:
        return metrics[fallback]
    return 0.0


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Experiment",
        "Attention",
        "Illumination",
        "SSL",
        "GRL",
        "MMD",
        "Consistency",
        "BestEpoch",
        "mAP50",
        "mAP75",
        "mAP50_95",
        "Precision",
        "Recall",
        "F1",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = selected_experiments(args.only)
    rows = []

    print("=" * 78)
    print("SICDA ABLATION STUDY: LLVIP -> FLIR")
    print("=" * 78)
    print(f"Experiments : {len(experiments)}")
    print(f"Epochs      : {args.epochs}")
    print(f"Seed        : {args.seed}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Output root : {output_root}")
    print("All experiments use the same seed/data protocol.")
    print("=" * 78)

    for name, setting in experiments:
        result = run_one(name, setting, args, output_root)
        metrics = result.get("best_metrics") or result.get("last_metrics") or {}
        rows.append(
            {
                "Experiment": name,
                "Attention": int(setting["attention"]),
                "Illumination": int(setting["illumination"]),
                "SSL": int(setting["ssl"]),
                "GRL": int(setting["domain"]),
                "MMD": int(setting["mmd"]),
                "Consistency": int(setting["consistency"]),
                "BestEpoch": int(result.get("best_epoch", 0)),
                "mAP50": metric_value(metrics, "mAP50"),
                "mAP75": metric_value(metrics, "mAP75"),
                "mAP50_95": metric_value(metrics, "mAP50_95", "mAP"),
                "Precision": metric_value(metrics, "Precision"),
                "Recall": metric_value(metrics, "Recall"),
                "F1": metric_value(metrics, "F1-score", "F1"),
            }
        )
        # Rewrite after every successful run so partial progress is preserved.
        write_csv(rows, output_root / "ablation_results.csv")

    print("\n" + "=" * 78)
    print("Ablation study completed successfully")
    print(f"Results: {output_root / 'ablation_results.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
