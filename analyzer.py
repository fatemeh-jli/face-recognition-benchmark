"""
analyzer.py — Cross-model comparison and ranking for the benchmark UI.

ModelStats accumulates raw per-frame data for every model and produces
a ranked summary dict that the dashboard and report both consume.

Design notes
------------
* Rankings are computed on every ``summarize()`` call so they always
  reflect the current data — no stale state.
* Ranking uses a stable sort so ties are broken by insertion order
  (i.e. the model that ran first wins ties), giving reproducible results.
* The summary dict schema is documented below so dashboard.py and
  report.py don't need to guess key names.

Summary dict schema (per model key)
------------------------------------
{
    "avg_fps"         : float,
    "avg_cpu"         : float,   # percent
    "avg_ram"         : float,   # MiB
    "avg_inference"   : float,   # ms
    "avg_confidence"  : float,   # percent (0–100)
    "n_frames"        : int,
    # ranking flags (exactly one model per flag is True)
    "fastest"         : bool,
    "most_accurate"   : bool,
    "lowest_cpu"      : bool,
    "lowest_ram"      : bool,
}
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class ModelStats:
    """
    Accumulates per-frame data across all active models and computes
    a ranked cross-model summary on demand.
    """

    def __init__(self) -> None:
        # { model_name: { metric: [values] } }
        self._data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        model: str,
        fps: float,
        cpu: float,
        ram: float,
        inference_ms: float,
        confidence: float,
    ) -> None:
        """Append one frame's measurements for *model*."""
        bucket = self._data[model]
        bucket["fps"].append(fps)
        bucket["cpu"].append(cpu)
        bucket["ram"].append(ram)
        bucket["inference"].append(inference_ms)
        bucket["confidence"].append(confidence)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._data.clear()

    # ------------------------------------------------------------------
    # Summary & ranking
    # ------------------------------------------------------------------

    def summarize(self) -> dict[str, dict[str, Any]]:
        """
        Return a ranked summary dict keyed by model name.

        Models with no data are omitted. Ranking flags are set only
        when at least one model has data.
        """
        summary: dict[str, dict[str, Any]] = {}

        for model, bucket in self._data.items():
            if not bucket["fps"]:
                continue
            summary[model] = {
                "avg_fps":        _safe_mean(bucket["fps"]),
                "avg_cpu":        _safe_mean(bucket["cpu"]),
                "avg_ram":        _safe_mean(bucket["ram"]),
                "avg_inference":  _safe_mean(bucket["inference"]),
                "avg_confidence": _safe_mean(bucket["confidence"]),
                "n_frames":       len(bucket["fps"]),
                # ranking flags — filled below
                "fastest":       False,
                "most_accurate": False,
                "lowest_cpu":    False,
                "lowest_ram":    False,
            }

        if not summary:
            return summary

        try:
            _tag(summary, "fastest",       key="avg_fps",        higher_is_better=True)
            _tag(summary, "most_accurate", key="avg_confidence",  higher_is_better=True)
            _tag(summary, "lowest_cpu",    key="avg_cpu",         higher_is_better=False)
            _tag(summary, "lowest_ram",    key="avg_ram",         higher_is_better=False)
        except Exception as exc:
            log.warning("Ranking calculation failed: %s", exc)

        return summary

    @property
    def model_names(self) -> list[str]:
        return list(self._data.keys())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _tag(
    summary: dict[str, dict[str, Any]],
    flag: str,
    key: str,
    higher_is_better: bool,
) -> None:
    """Set ``flag`` to True for the best model according to *key*."""
    best = (max if higher_is_better else min)(
        summary, key=lambda m: summary[m][key]
    )
    summary[best][flag] = True