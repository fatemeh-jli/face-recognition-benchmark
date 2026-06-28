"""
metrics.py — Per-frame telemetry collection and system-resource sampling.

MetricsTracker  : rolling accumulator for one model's live stats.
get_system_usage: lightweight CPU / RAM snapshot (non-blocking).

Design notes
------------
* psutil.cpu_percent(interval=None) is non-blocking; it returns the usage
  since the *last* call, so calling it once per frame is exactly right.
  Never pass interval > 0 inside the video loop — it would block.
* RAM is reported in MiB (mebibytes), consistent with most system monitors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import psutil

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System resource helpers
# ---------------------------------------------------------------------------

def get_system_usage() -> tuple[float, float]:
    """
    Return (cpu_percent, ram_mib).

    cpu_percent : 0–100, averaged across all logical cores since last call.
    ram_mib     : process RSS in mebibytes.
    """
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / (1024 ** 2)
    return cpu, ram


# ---------------------------------------------------------------------------
# Per-model rolling accumulator
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    """Internal storage for one metric series."""
    values: list[float] = field(default_factory=list)

    def push(self, v: float) -> None:
        self.values.append(v)

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def std(self) -> float:
        return float(np.std(self.values)) if len(self.values) > 1 else 0.0

    def minimum(self) -> float:
        return float(np.min(self.values)) if self.values else 0.0

    def maximum(self) -> float:
        return float(np.max(self.values)) if self.values else 0.0

    def count(self) -> int:
        return len(self.values)

    def clear(self) -> None:
        self.values.clear()


class MetricsTracker:
    """
    Accumulates per-frame performance data for a single model.

    Call ``update(...)`` once per frame.
    Call ``summary()`` to retrieve descriptive statistics.
    Call ``reset()`` to clear all accumulated data.
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self._fps         = _Bucket()
        self._inference   = _Bucket()
        self._cpu         = _Bucket()
        self._ram         = _Bucket()
        self._confidence  = _Bucket()
        self._face_count  = _Bucket()

    def update(
        self,
        *,
        inference_time: float,
        fps: float,
        cpu: float,
        ram: float,
        confidence: float,
        face_count: int,
    ) -> None:
        """Record one frame's metrics. All keyword-only to prevent argument ordering bugs."""
        self._fps.push(fps)
        self._inference.push(inference_time)
        self._cpu.push(cpu)
        self._ram.push(ram)
        self._confidence.push(confidence)
        self._face_count.push(float(face_count))

    def summary(self) -> dict[str, float]:
        """
        Return a flat dict of descriptive statistics.

        Keys follow the pattern  <metric>_<stat>, e.g. ``fps_mean``.
        """
        return {
            # FPS
            "fps_mean":        self._fps.mean(),
            "fps_std":         self._fps.std(),
            "fps_min":         self._fps.minimum(),
            "fps_max":         self._fps.maximum(),
            # Inference latency (ms)
            "inference_mean":  self._inference.mean(),
            "inference_std":   self._inference.std(),
            "inference_min":   self._inference.minimum(),
            "inference_max":   self._inference.maximum(),
            # System load
            "cpu_mean":        self._cpu.mean(),
            "ram_mean":        self._ram.mean(),
            # Recognition quality
            "confidence_mean": self._confidence.mean(),
            "confidence_std":  self._confidence.std(),
            # Detection
            "face_count_mean": self._face_count.mean(),
            # Sample size (useful for weighting in reports)
            "n_frames":        float(self._fps.count()),
        }

    @property
    def frame_count(self) -> int:
        return self._fps.count()
    
    # Add to metrics.py
from sklearn.metrics import roc_curve, auc

class BiometricEvaluator:
    """
    Computes ISO/IEC 19795-1 biometric evaluation metrics.
    Call add_pair() for every genuine/impostor comparison, then compute().
    """
    def __init__(self):
        self.scores: list[float] = []
        self.labels: list[int] = []   # 1 = genuine pair, 0 = impostor

    def add_pair(self, score: float, is_genuine: bool):
        self.scores.append(score)
        self.labels.append(int(is_genuine))

    def compute(self) -> dict:
        scores = np.array(self.scores)
        labels = np.array(self.labels)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # EER: point where FAR == FRR
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

        # FAR/FRR at operating threshold (e.g. 0.55 cosine)
        op_thresh = 0.55
        predicted = (scores >= op_thresh).astype(int)
        FAR = np.mean(predicted[labels == 0])   # impostor accepted
        FRR = np.mean(1 - predicted[labels == 1])  # genuine rejected

        return {
            "AUC":       round(float(roc_auc), 4),
            "EER":       round(float(eer), 4),
            "FAR":       round(float(FAR), 4),
            "FRR":       round(float(FRR), 4),
            "fpr":       fpr.tolist(),      # for ROC curve plotting
            "tpr":       tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }