"""
report.py — Research-grade benchmark report generator.

Outputs (all written to ``output_dir``):
  benchmark_results.txt   — human-readable summary with ranking table
  benchmark_raw.csv       — full per-frame log for statistical analysis
  benchmark_summary.json  — machine-readable summary for reproducibility

Design notes
------------
* The .txt report uses a fixed-width table so it pastes cleanly into
  LaTeX verbatim blocks or a thesis appendix.
* The .csv includes a Unix timestamp column so frame data can be
  aligned on a time axis in R / Python / Excel.
* The .json summary mirrors the .txt ranking logic so downstream
  scripts can consume it without re-parsing plain text.
* numpy is used only for aggregation; no runtime dependency beyond the
  standard library + numpy.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key mapping: history dict key → display label
# ---------------------------------------------------------------------------
_METRIC_KEYS = {
    "fps":           "Avg FPS",
    "inference_ms":  "Avg Inference (ms)",
    "cpu":           "Avg CPU (%)",
    "ram":           "Avg RAM (MiB)",
    "confidence":    "Avg Confidence (%)",
}

# History records written by main.py use "inference_ms"; the original code
# used "inference". We normalise both at load time.
_KEY_ALIASES = {"inference": "inference_ms"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    history: list[dict[str, Any]],
    output_dir: str | Path = ".",
) -> None:
    """
    Generate all three report artefacts from *history*.

    Parameters
    ----------
    history   : list of per-frame dicts produced by the main benchmark loop.
    output_dir: directory where output files are written (created if absent).
    """
    if not history:
        log.warning("No benchmark data collected — report skipped.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Normalise key aliases so both old and new history formats work
    history = [_normalise_record(r) for r in history]

    per_model = _aggregate(history)

    if not per_model:
        log.warning("Aggregation produced no data — report skipped.")
        return

    summary  = _compute_summary(per_model)
    rankings = _rank(summary)

    _write_text(summary, rankings, out / "benchmark_results.txt")
    _write_csv(history,            out / "benchmark_raw.csv")
    _write_json(summary, rankings, out / "benchmark_summary.json")

    log.info("Report written to: %s", out.resolve())


# ---------------------------------------------------------------------------
# Internal helpers — aggregation
# ---------------------------------------------------------------------------

def _normalise_record(r: dict) -> dict:
    """Replace key aliases so downstream code can always use canonical names."""
    out = {}
    for k, v in r.items():
        out[_KEY_ALIASES.get(k, k)] = v
    return out


def _aggregate(history: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Group raw frame records by model and collect metric lists."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for record in history:
        model = record.get("model", "Unknown")
        if model not in buckets:
            buckets[model] = {k: [] for k in _METRIC_KEYS}

        for metric_key in _METRIC_KEYS:
            value = record.get(metric_key)
            if value is not None:
                buckets[model][metric_key].append(float(value))

    return buckets


def _compute_summary(
    per_model: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    """Compute mean ± std for every metric, per model."""
    summary: dict[str, dict[str, float]] = {}
    for model, metrics in per_model.items():
        entry: dict[str, float] = {"n_frames": 0.0}
        for key, values in metrics.items():
            if values:
                entry[f"{key}_mean"] = float(np.mean(values))
                entry[f"{key}_std"]  = float(np.std(values))
                entry["n_frames"]    = float(len(values))
            else:
                entry[f"{key}_mean"] = 0.0
                entry[f"{key}_std"]  = 0.0
        summary[model] = entry
    return summary


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def _rank(summary: dict[str, dict[str, float]]) -> dict[str, str]:
    """Return a dict mapping rank labels to the winning model name."""
    if not summary:
        return {}

    def best(key: str, higher: bool) -> str:
        fn = max if higher else min
        return fn(summary, key=lambda m: summary[m].get(key, 0.0))

    return {
        "fastest":       best("fps_mean",        higher=True),
        "most_accurate": best("confidence_mean",  higher=True),
        "lowest_cpu":    best("cpu_mean",         higher=False),
        "lowest_ram":    best("ram_mean",         higher=False),
        "lowest_latency":best("inference_ms_mean",higher=False),
        "recommended":   _recommend(summary),
    }


def _recommend(summary: dict[str, dict[str, float]]) -> str:
    """
    Composite recommendation: weighted score across FPS, confidence,
    and inverse CPU.  Weights are equal (⅓ each) and each metric is
    min-max normalised to [0, 1] so units don't dominate.
    """
    models = list(summary.keys())
    if len(models) == 1:
        return models[0]

    def column(key: str) -> np.ndarray:
        return np.array([summary[m].get(key, 0.0) for m in models], dtype=float)

    def norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    fps_score  = norm(column("fps_mean"))
    conf_score = norm(column("confidence_mean"))
    cpu_score  = 1.0 - norm(column("cpu_mean"))    # lower CPU → better

    composite  = (fps_score + conf_score + cpu_score) / 3.0
    return models[int(np.argmax(composite))]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_text(
    summary: dict[str, dict[str, float]],
    rankings: dict[str, str],
    path: Path,
) -> None:
    sep   = "=" * 66
    model_names = list(summary.keys())

    lines: list[str] = [
        sep,
        "  FACE RECOGNITION BENCHMARK REPORT",
        f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "  PER-MODEL STATISTICS  (mean ± std)",
        "",
    ]

    # Fixed-width table
    col_w = 18
    header_row = f"  {'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in model_names)
    lines.append(header_row)
    lines.append("  " + "-" * (22 + col_w * len(model_names)))

    metric_rows = [
        ("FPS",              "fps_mean",           "fps_std"),
        ("Inference (ms)",   "inference_ms_mean",  "inference_ms_std"),
        ("CPU (%)",          "cpu_mean",            "cpu_std"),
        ("RAM (MiB)",        "ram_mean",            "ram_std"),
        ("Confidence (%)",   "confidence_mean",     "confidence_std"),
        ("Frames sampled",   "n_frames",            None),
    ]

    for label, mean_key, std_key in metric_rows:
        row = f"  {label:<22}"
        for m in model_names:
            mean = summary[m].get(mean_key, 0.0)
            if std_key:
                std = summary[m].get(std_key, 0.0)
                cell = f"{mean:.2f}±{std:.2f}"
            else:
                cell = f"{int(mean)}"
            row += f"{cell:>{col_w}}"
        lines.append(row)

    lines += [
        "",
        sep,
        "  RANKINGS",
        sep,
        "",
        f"  Fastest model     : {rankings.get('fastest',       '—')}",
        f"  Most accurate     : {rankings.get('most_accurate', '—')}",
        f"  Lowest CPU usage  : {rankings.get('lowest_cpu',    '—')}",
        f"  Lowest RAM usage  : {rankings.get('lowest_ram',    '—')}",
        f"  Lowest latency    : {rankings.get('lowest_latency','—')}",
        "",
        sep,
        "  RECOMMENDATION (composite: FPS + Accuracy + CPU efficiency)",
        sep,
        "",
        f"  ► {rankings.get('recommended', '—')}",
        "",
        sep,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Text report  → %s", path)


def _write_csv(history: list[dict], path: Path) -> None:
    if not history:
        return

    # Build a stable ordered field list from all keys seen in history
    all_keys: list[str] = []
    seen: set[str] = set()
    for record in history:
        for k in record:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    log.info("CSV log       → %s  (%d rows)", path, len(history))


def _write_json(
    summary: dict[str, dict[str, float]],
    rankings: dict[str, str],
    path: Path,
) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary":      summary,
        "rankings":     rankings,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("JSON summary  → %s", path)
    
    def write_latex_table(summary: dict, path: Path) -> None:
        """
    Emit a LaTeX table for direct inclusion in a thesis.
    Compatible with booktabs (\\toprule, \\midrule, \\bottomrule).
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Face Recognition Model Benchmark Results}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & AUC & EER (\%) & FAR (\%) & FRR (\%) & FPS (mean) \\",
        r"\midrule",
    ]
    for model, s in summary.items():
        lines.append(
            f"{model} & {s['AUC']:.4f} & {s['EER']*100:.2f} & "
            f"{s['FAR']*100:.2f} & {s['FRR']*100:.2f} & {s['fps_mean']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))