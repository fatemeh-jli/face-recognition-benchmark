"""
Face Recognition Benchmark System
===================================
A research-grade benchmarking tool for comparing face recognition models
(InsightFace/Buffalo_S, MobileFaceNet, FaceNet) in real-time.

Outputs: CSV logs, JSON summaries, and a final PDF-friendly report.
Suitable for academic benchmarking and thesis submission.

Controls:
    1 → InsightFace (Buffalo_S)
    2 → MobileFaceNet (ONNX)
    3 → FaceNet (ONNX)
    F → Freeze frame
    U → Unfreeze frame
    S → Save snapshot
    C → Clear metrics
    Q → Quit
"""

import cv2
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from insightface.app import FaceAnalysis

# Project-local modules
from recognition import recognize_face
from metrics import MetricsTracker, get_system_usage
from analyzer import ModelStats
from dashboard import build_dashboard
from report import generate_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_NAME = "Face Recognition Benchmark System"

# Camera frame dimensions fed into the display (does NOT affect model input)
DISPLAY_CAMERA_W, DISPLAY_CAMERA_H = 900, 600
DISPLAY_DASHBOARD_W, DISPLAY_DASHBOARD_H = 500, 600

# InsightFace detection resolution — bump to 640 for accuracy, 320 for speed
INSIGHT_DET_SIZE = (640, 640)


ARCFACE_PATH = Path("w600k_r100.onnx")
ADAFACE_PATH = Path("adaface_ir18_vgg2.onnx")
SFACE_PATH   = Path("face_recognition_sface_2021dec.onnx")

# Haar cascade for ONNX-based pipelines
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Snapshot output directory
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Supported model keys
MODEL_KEYS = ("ARCFACE", "ADAFACE", "SFACE")

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """Stores per-frame inference results."""
    name: str = "Unknown"
    confidence: float = 0.0
    face_count: int = 0
    inference_ms: float = 0.0
    fps: float = 0.0
    cpu: float = 0.0
    ram: float = 0.0


@dataclass
class AppState:
    """Mutable runtime state for the benchmark loop."""
    model_mode: str = "INSIGHTFACE"
    frozen: bool = False
    current_frame: Optional[np.ndarray] = None
    snapshot_id: int = 0
    history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
def load_insightface(det_size: tuple) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")  # ArcFace R100 backbone
    app.prepare(ctx_id=0, det_size=det_size)  # GPU
    return app


def load_onnx_session(path: Path, name: str) -> Optional[ort.InferenceSession]:
    """Return an ONNX InferenceSession or None if the file is missing."""
    if not path.exists():
        log.warning("%s not found — running in simulation mode for %s.", path, name)
        return None
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    log.info("Loaded %s (%s).", name, path)
    return session


# ---------------------------------------------------------------------------
# Pre-processing helpers (moved OUTSIDE the loop)
# ---------------------------------------------------------------------------
def preprocess_mobilefacenet(face_img: np.ndarray) -> np.ndarray:
    """Resize to 112×112, normalise to [0, 1], NCHW layout."""
    blob = cv2.resize(face_img, (112, 112)).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)


def preprocess_facenet(face_img: np.ndarray) -> np.ndarray:
    """Resize to 160×160, normalise to [-1, 1], NCHW layout."""
    blob = (cv2.resize(face_img, (160, 160)).astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)


def simulate_embedding(model_mode: str) -> np.ndarray:
    """Return a random unit-normalised embedding for demo / missing-model fallback."""
    dim = 512 if model_mode == "FACENET" else 128
    vec = np.random.rand(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


# ---------------------------------------------------------------------------
# Inference pipelines
# ---------------------------------------------------------------------------
def run_insightface(
    insight: FaceAnalysis,
    frame: np.ndarray,
) -> tuple[np.ndarray, str, float, int]:
    """
    Run InsightFace pipeline.

    Returns
    -------
    annotated_frame, identity_name, confidence, face_count
    """
    results = insight.get(frame)
    name, conf, face_count = "Unknown", 0.0, len(results)

    for r in results:
        if r.embedding is None:
            continue
        x1, y1, x2, y2 = map(int, r.bbox)
        name, conf = recognize_face(r.embedding)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{name} {conf:.1f}%", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    return frame, name, conf, face_count


def run_onnx_pipeline(
    model_mode: str,
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    mobilefacenet_sess: Optional[ort.InferenceSession],
    facenet_sess: Optional[ort.InferenceSession],
) -> tuple[np.ndarray, str, float, int]:
    """
    Run Haar + ONNX embedding pipeline for MobileFaceNet or FaceNet.

    Returns
    -------
    annotated_frame, identity_name, confidence, face_count
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    name, conf, face_count = "Unknown", 0.0, len(detected)

    for (x, y, w, h) in detected:
        face_img = frame[y: y + h, x: x + w]
        if face_img.size == 0:
            continue

        emb = _extract_embedding(model_mode, face_img, mobilefacenet_sess, facenet_sess)
        name, conf = recognize_face(emb)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame, f"{name} {conf:.1f}%", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
        )

    return frame, name, conf, face_count


def _extract_embedding(
    model_mode: str,
    face_img: np.ndarray,
    mobilefacenet_sess: Optional[ort.InferenceSession],
    facenet_sess: Optional[ort.InferenceSession],
) -> np.ndarray:
    """Route to the correct ONNX session or fall back to simulation."""
    if model_mode == "MOBILEFACENET" and mobilefacenet_sess is not None:
        blob = preprocess_mobilefacenet(face_img)
        input_name = mobilefacenet_sess.get_inputs()[0].name
        return mobilefacenet_sess.run(None, {input_name: blob})[0][0]

    if model_mode == "FACENET" and facenet_sess is not None:
        blob = preprocess_facenet(face_img)
        input_name = facenet_sess.get_inputs()[0].name
        return facenet_sess.run(None, {input_name: blob})[0][0]

    # Simulation fallback (ONNX file missing)
    return simulate_embedding(model_mode)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_stable_fps(prev_time: float) -> tuple[float, float]:
    """
    Return (fps, elapsed_ms) using a monotonic wall-clock delta.

    Using time.perf_counter() instead of time.time() gives better
    sub-millisecond resolution on all platforms.
    """
    now = time.perf_counter()
    elapsed = now - prev_time
    fps = 1.0 / (elapsed + 1e-9)
    return fps, elapsed * 1000.0, now


# ---------------------------------------------------------------------------
# Camera helper
# ---------------------------------------------------------------------------
def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {index}.")
    # Hint the OS to use a small internal buffer to reduce latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark() -> None:  # noqa: C901  (complexity accepted for a monolithic loop)
    # --- Load models ---
    insight = load_insightface(INSIGHT_DET_SIZE)
    arcface_sess  = load_onnx_session(ARCFACE_PATH,  "ArcFace")
    adaface_sess  = load_onnx_session(ADAFACE_PATH,  "AdaFace")
    sface_sess    = load_onnx_session(SFACE_PATH,    "SFace")

    # Build the Haar cascade ONCE (heavy XML parse)
    haar_detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if haar_detector.empty():
        raise RuntimeError(f"Haar cascade not found at: {HAAR_CASCADE}")

    # --- Init metrics ---
    metrics: dict[str, MetricsTracker] = {k: MetricsTracker() for k in MODEL_KEYS}
    model_stats = ModelStats()
    state = AppState()

    cap = open_camera(0)
    log.info("System running. Keys: 1/2/3 switch model | F freeze | U unfreeze | S snapshot | C clear | Q quit")

    # Warm-up timing reference
    loop_prev_time = time.perf_counter()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        # --- Frame acquisition ---
        if not state.frozen:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame — retrying…")
                continue
            state.current_frame = frame  # keep a reference without copying
        else:
            frame = state.current_frame

        # Mirror for natural user experience
        frame = cv2.flip(frame, 1)

        # --- Inference ---
        t_start = time.perf_counter()


        if state.model_mode == "ARCFACE":
           frame, name, conf, face_count = run_insightface(insight, frame)
        else:
           frame, name, conf, face_count = run_onnx_pipeline(
             state.model_mode, frame, haar_detector,
             adaface_sess, sface_sess,
        )

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        # FPS based on full loop wall-clock (more representative than inference alone)
        fps, _, loop_prev_time = compute_stable_fps(loop_prev_time)

        cpu, ram = get_system_usage()

        # --- Metrics update ---
        metrics[state.model_mode].update(
            inference_time=inference_ms,
            fps=fps,
            cpu=cpu,
            ram=ram,
            confidence=conf,
            face_count=face_count,
        )

        state.history.append({
            "model": state.model_mode,
            "fps": round(fps, 2),
            "cpu": round(cpu, 2),
            "ram": round(ram, 2),
            "inference_ms": round(inference_ms, 3),
            "confidence": round(conf, 2),
            "face_count": face_count,
            "timestamp": time.time(),
        })

        model_stats.add(state.model_mode, fps, cpu, ram, inference_ms, conf)
        stats = model_stats.summarize()

        # --- UI rendering ---
        dashboard = build_dashboard(
            model_name=state.model_mode,
            frozen=state.frozen,
            identity=name,
            score=conf,
            fps=fps,
            cpu=cpu,
            ram=ram,
            inference_ms=inference_ms,
            face_count=face_count,
            stats=stats,
        )

        frame_display = cv2.resize(frame, (DISPLAY_CAMERA_W, DISPLAY_CAMERA_H))
        dashboard_display = cv2.resize(dashboard, (DISPLAY_DASHBOARD_W, DISPLAY_DASHBOARD_H))
        combined = np.hstack((frame_display, dashboard_display))
        cv2.imshow(WINDOW_NAME, combined)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            state.model_mode = "ARCFACE" 
        elif key == ord("2"):
            state.model_mode = "ADAFACE"
        elif key == ord("3"):
            state.model_mode = "SFACE" 
        elif key == ord("f"):
            state.frozen = True
            log.info("Frame frozen")
        elif key == ord("u"):
            state.frozen = False
            log.info("Frame unfrozen")
        elif key == ord("s"):
            state.snapshot_id += 1
            snap_path = SNAPSHOT_DIR / f"frame_{state.snapshot_id:04d}.jpg"
            cv2.imwrite(str(snap_path), frame)
            log.info("Snapshot saved: %s", snap_path)
        elif key == ord("c"):
            state.history.clear()
            model_stats = ModelStats()
            for tracker in metrics.values():
                tracker.reset()  # assumes MetricsTracker exposes reset()
            log.info("Metrics cleared")

    # --- Teardown ---
    cap.release()
    cv2.destroyAllWindows()
    log.info("Camera released.")

    # Final report
    try:
        generate_report(state.history)
        log.info("Report generated.")
    except Exception as exc:
        log.error("Report generation failed: %s", exc)

    log.info("System closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_benchmark()
