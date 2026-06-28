"""
dashboard.py — Real-time OpenCV sidebar panel renderer.

``build_dashboard`` returns a (600, 500, 3) uint8 BGR image that is
horizontally stacked with the camera frame in main.py.

Design principles
-----------------
* All drawing calls use pre-computed coordinates — no arithmetic inside
  loops that repeat every frame.
* Colors follow a consistent dark-terminal palette; accent colour
  distinguishes section headers from data.
* Badge glyphs (✓) are replaced with ASCII [F] / [A] / [C] / [R] tags
  because OpenCV's default font does not support Unicode.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette  (BGR)
# ---------------------------------------------------------------------------
_BG     = (18,  18,  18)
_CARD   = (28,  28,  28)
_BORDER = (50,  50,  50)
_TEXT   = (230, 230, 230)
_DIM    = (140, 140, 140)
_ACCENT = (255, 200,   0)   # gold — section headers
_GREEN  = (80,  220,  80)   # good values / identity match
_RED    = (80,   80, 220)   # warning / frozen
_CYAN   = (255, 200,   0)   # match score
_YELLOW = (0,   200, 200)

# ---------------------------------------------------------------------------
# Font shorthand
# ---------------------------------------------------------------------------
_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_SOLID = cv2.LINE_AA


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _card(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draw a filled rounded-rectangle card (flat, since OpenCV lacks rounded rects)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), _CARD,   -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _BORDER,  1)


def _text(
    img: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple = _TEXT,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    cv2.putText(img, label, (x, y), _FONT, scale, color, thickness, _SOLID)


def _header(img: np.ndarray, label: str, x: int, y: int) -> None:
    _text(img, label, x, y, _ACCENT, 0.52, 1)
    # thin underline
    text_w, _ = cv2.getTextSize(label, _FONT, 0.52, 1)[0]
    cv2.line(img, (x, y + 4), (x + text_w, y + 4), _ACCENT, 1, _SOLID)


def _bar(
    img: np.ndarray,
    value: float,
    max_val: float,
    x: int,
    y: int,
    w: int = 200,
    h: int = 6,
    color: tuple = _GREEN,
) -> None:
    """Draw a horizontal progress bar."""
    ratio = max(0.0, min(1.0, value / (max_val + 1e-9)))
    cv2.rectangle(img, (x, y), (x + w, y + h), _BORDER, -1)
    cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_status(
    img: np.ndarray,
    model_name: str,
    frozen: bool,
    identity: str,
    score: float,
) -> None:
    _card(img, 20, 10, 460, 140)
    _header(img, "SYSTEM STATUS", 35, 38)

    frozen_color = _RED if frozen else _GREEN
    frozen_label = "FROZEN" if frozen else "LIVE"

    _text(img, f"Model : {model_name}",        35,  70)
    _text(img, f"Stream: {frozen_label}",       35,  95, frozen_color)
    _text(img, f"ID    : {identity}",           35, 120)
    _text(img, f"Score : {score:5.1f}%",       270, 120, _YELLOW)


def _section_benchmark(
    img: np.ndarray,
    fps: float,
    cpu: float,
    ram: float,
    inference_ms: float,
    face_count: int,
) -> None:
    _card(img, 20, 165, 460, 210)
    _header(img, "LIVE BENCHMARK", 35, 193)

    # FPS — colour-coded: green ≥ 20, yellow ≥ 10, red < 10
    fps_color = _GREEN if fps >= 20 else (_YELLOW if fps >= 10 else _RED)
    _text(img, f"FPS       : {fps:5.1f}",     35, 225, fps_color)
    _bar(img, fps, 60, 230, 230, color=fps_color)

    _text(img, f"Inference : {inference_ms:6.1f} ms", 35, 258)
    _bar(img, inference_ms, 500, 230, 263)

    _text(img, f"CPU       : {cpu:5.1f}%",    35, 291)
    _bar(img, cpu, 100, 230, 296, color=_YELLOW)

    _text(img, f"RAM       : {ram:6.0f} MiB", 35, 324)
    _bar(img, ram, 8192, 230, 329, color=_CYAN)

    _text(img, f"Faces     : {face_count}",   35, 357, _DIM)


def _section_comparison(
    img: np.ndarray,
    stats: dict[str, dict[str, Any]],
) -> None:
    _card(img, 20, 390, 460, 200)
    _header(img, "MODEL COMPARISON", 35, 418)

    # Column headers
    _text(img, "Model",      35,  445, _DIM, 0.40)
    _text(img, "FPS",       200,  445, _DIM, 0.40)
    _text(img, "Inf(ms)",   275,  445, _DIM, 0.40)
    _text(img, "Conf%",     370,  445, _DIM, 0.40)
    _text(img, "Badges",    435,  445, _DIM, 0.40)

    # Display names map (internal key → short display label)
    display = {
        "INSIGHTFACE":   "Buffalo_S",
        "MOBILEFACENET": "MobFaceNet",
        "FACENET":       "FaceNet",
    }

    badge_map = {
        "fastest":       "F",
        "most_accurate": "A",
        "lowest_cpu":    "C",
        "lowest_ram":    "R",
    }

    y = 468
    for internal_key, short_name in display.items():
        s = stats.get(internal_key, {})

        badges = "".join(
            f"[{glyph}]"
            for flag, glyph in badge_map.items()
            if s.get(flag, False)
        )

        row_color = _GREEN if badges else _TEXT

        _text(img, short_name,                          35,  y, row_color, 0.42)
        _text(img, f"{s.get('avg_fps', 0):4.1f}",     200,  y, row_color, 0.42)
        _text(img, f"{s.get('avg_inference', 0):5.1f}", 275, y, row_color, 0.42)
        _text(img, f"{s.get('avg_confidence', 0):4.1f}", 370, y, row_color, 0.42)
        _text(img, badges or "—",                      430,  y, _ACCENT,   0.40)

        y += 36


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dashboard(
    *,
    model_name: str,
    frozen: bool,
    identity: str,
    score: float,
    fps: float,
    cpu: float,
    ram: float,
    inference_ms: float,
    face_count: int,
    stats: dict[str, dict[str, Any]],
) -> np.ndarray:
    """
    Render the sidebar dashboard and return a (600 × 500 × 3) BGR image.

    All parameters are keyword-only to prevent silent argument-order bugs
    at call sites.
    """
    canvas = np.full((600, 500, 3), _BG, dtype=np.uint8)

    _section_status(canvas, model_name, frozen, identity, score)
    _section_benchmark(canvas, fps, cpu, ram, inference_ms, face_count)
    _section_comparison(canvas, stats)

    # Footer
    _text(canvas, "1:InsightFace  2:MobFaceNet  3:FaceNet", 22, 590, _DIM, 0.36)

    return canvas