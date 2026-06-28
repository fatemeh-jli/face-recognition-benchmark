"""
recognition.py — Face identity lookup via cosine similarity.

The database is a flat directory of .npy files, one per person.
File name (without extension) is treated as the identity label.

Each .npy file may contain:
  - A single embedding  : shape (D,)
  - Multiple embeddings : shape (N, D)  — all are compared; best match wins.

Usage:
    from recognition import FaceDatabase
    db  = FaceDatabase("database")
    name, confidence = db.recognize(embedding)
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class FaceDatabase:
    """
    Loads .npy embedding files from ``db_path`` and provides
    cosine-similarity-based identity lookup.

    Parameters
    ----------
    db_path : str | Path
        Directory containing one .npy file per identity.
    threshold : float
        Minimum cosine similarity (0–1) to accept a match.
        Scores below this return "Unknown".
    """

    def __init__(self, db_path: str | Path = "database", threshold: float = 0.55) -> None:
        self.threshold = threshold
        self._embeddings: list[np.ndarray] = []   # shape (D,) each
        self._names: list[str] = []

        self._load(Path(db_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, query: np.ndarray) -> tuple[str, float]:
        """
        Return (identity_name, confidence_percent) for *query*.

        confidence_percent is best_cosine_score × 100, regardless of
        whether the threshold is met (so the caller can always log it).
        """
        if len(self._embeddings) == 0:
            return "Unknown", 0.0

        query_norm = _unit(query)
        best_score, best_name = -1.0, "Unknown"

        for emb, name in zip(self._embeddings, self._names):
            score = float(np.dot(query_norm, emb))   # emb already unit-normalised
            if score > best_score:
                best_score, best_name = score, name

        confidence = best_score * 100.0
        identity = best_name if best_score >= self.threshold else "Unknown"
        return identity, confidence

    def __len__(self) -> int:
        return len(self._names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, db_path: Path) -> None:
        if not db_path.exists():
            log.warning("Database directory '%s' not found — recognition disabled.", db_path)
            return

        files = sorted(db_path.glob("*.npy"))
        if not files:
            log.warning("No .npy files found in '%s'.", db_path)
            return

        for f in files:
            try:
                data = np.load(str(f)).astype(np.float32)
                name = f.stem                          # filename without extension

                # Support both single (D,) and multi-embedding (N, D) files
                if data.ndim == 1:
                    self._embeddings.append(_unit(data))
                    self._names.append(name)
                elif data.ndim == 2:
                    for row in data:
                        self._embeddings.append(_unit(row))
                        self._names.append(name)
                else:
                    log.warning("Skipping '%s': unexpected shape %s.", f.name, data.shape)
            except Exception as exc:
                log.error("Could not load '%s': %s", f, exc)

        log.info("Database loaded: %d embedding(s) for %d identity/identities.",
                 len(self._embeddings), len(set(self._names)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return a unit-normalised copy of vector *v*."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Module-level singleton + legacy shim
# ---------------------------------------------------------------------------
# A single shared database instance so other modules can do:
#   from recognition import recognize_face
# while new code can do:
#   from recognition import FaceDatabase

_default_db: FaceDatabase | None = None


def _get_db() -> FaceDatabase:
    global _default_db
    if _default_db is None:
        _default_db = FaceDatabase()
    return _default_db


def recognize_face(query_embedding: np.ndarray, threshold: float = 0.55) -> tuple[str, float]:
    """Legacy functional shim — delegates to the module-level FaceDatabase."""
    db = _get_db()
    db.threshold = threshold
    return db.recognize(query_embedding)