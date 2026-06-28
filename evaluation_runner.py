"""
evaluation_runner.py — Offline, reproducible benchmark evaluation.

Runs all three models against a standard dataset (LFW or custom pairs)
and writes benchmark_summary.json + benchmark_raw.csv + LaTeX table.

This module is intentionally SEPARATE from the live webcam loop (main.py).
Separation of concerns:
  main.py             → real-time demo, UI, snapshots
  evaluation_runner.py → reproducible, dataset-based evaluation for your thesis

Supported dataset formats
--------------------------
LFW (standard):
  pairs.txt   — space-separated: name1 n1 name2 n2  (impostor)
                                  name  n1 n2        (genuine)
  images/     — one sub-folder per identity, images named <name>_NNNN.jpg

Custom pairs:
  pairs_custom.csv — columns: img1_path, img2_path, label (1=genuine, 0=impostor)

Usage
-----
    python evaluation_runner.py --dataset lfw --data_dir datasets/lfw
    python evaluation_runner.py --dataset custom --pairs pairs_custom.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

# Project modules
from recognition import FaceDatabase, _unit
from security import BiometricEvaluator, compute_anti_spoofing_metrics, SpoofSimulator

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Cosine similarity (kept local so evaluation_runner has no hidden deps)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [−1, 1] between two embedding vectors."""
    a = _unit(np.asarray(a, dtype=np.float32))
    b = _unit(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_lfw_pairs(
    data_dir: str | Path,
    pairs_file: str = "pairs.txt",
    max_pairs: int | None = None,
) -> list[tuple[Path, Path, int]]:
    """
    Parse the standard LFW pairs.txt format.

    File format (http://vis-www.cs.umass.edu/lfw/):
      Line 1  : N_folds  N_pairs_per_fold   (header — skipped)
      Genuine : name  img_id1  img_id2
      Impostor: name1 img_id1  name2  img_id2

    Returns
    -------
    list of (img1_path, img2_path, label)
        label = 1 for genuine pairs, 0 for impostor pairs.
    """
    data_dir = Path(data_dir)
    pairs_path = data_dir / pairs_file

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"LFW pairs file not found: {pairs_path}\n"
            "Download from http://vis-www.cs.umass.edu/lfw/pairs.txt"
        )

    pairs: list[tuple[Path, Path, int]] = []

    with open(pairs_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Skip header line
    data_lines = lines[1:]

    for line in data_lines:
        parts = line.split()
        try:
            if len(parts) == 3:
                # Genuine pair: name img_id1 img_id2
                name, id1, id2 = parts[0], int(parts[1]), int(parts[2])
                p1 = _lfw_img_path(data_dir, name, id1)
                p2 = _lfw_img_path(data_dir, name, id2)
                label = 1
            elif len(parts) == 4:
                # Impostor pair: name1 img_id1 name2 img_id2
                name1, id1, name2, id2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                p1 = _lfw_img_path(data_dir, name1, id1)
                p2 = _lfw_img_path(data_dir, name2, id2)
                label = 0
            else:
                log.warning("Skipping malformed line: %s", line)
                continue
        except (ValueError, IndexError) as exc:
            log.warning("Skipping line (parse error: %s): %s", exc, line)
            continue

        if p1.exists() and p2.exists():
            pairs.append((p1, p2, label))
        else:
            log.debug("Image not found, skipping: %s or %s", p1, p2)

        if max_pairs and len(pairs) >= max_pairs:
            break

    log.info("Loaded %d pairs (%d genuine, %d impostor).",
             len(pairs),
             sum(1 for _, _, l in pairs if l == 1),
             sum(1 for _, _, l in pairs if l == 0))

    return pairs


def load_custom_pairs(csv_path: str | Path) -> list[tuple[Path, Path, int]]:
    """
    Load pairs from a CSV with columns: img1_path, img2_path, label.
    label must be 1 (genuine) or 0 (impostor).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Custom pairs CSV not found: {csv_path}")

    pairs: list[tuple[Path, Path, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1 = Path(row["img1_path"])
            p2 = Path(row["img2_path"])
            label = int(row["label"])
            if p1.exists() and p2.exists():
                pairs.append((p1, p2, label))
            else:
                log.warning("Image not found, skipping: %s or %s", p1, p2)

    log.info("Loaded %d custom pairs.", len(pairs))
    return pairs


def _lfw_img_path(data_dir: Path, name: str, img_id: int) -> Path:
    """Resolve LFW image path: data_dir/<name>/<name>_NNNN.jpg"""
    fname = f"{name}_{img_id:04d}.jpg"
    return data_dir / name / fname


# ---------------------------------------------------------------------------
# Embedding extractor wrapper
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """
    Wraps InsightFace / ONNX sessions into a unified embed(img) interface
    for use in evaluation_runner without importing from main.py.

    Parameters
    ----------
    model_name : "INSIGHTFACE" | "MOBILEFACENET" | "FACENET"
    insight    : InsightFace FaceAnalysis instance (required for INSIGHTFACE)
    onnx_sess  : ONNX InferenceSession (required for MOBILEFACENET / FACENET)
    """

    def __init__(self, model_name: str, insight=None, onnx_sess=None) -> None:
        self.model_name = model_name.upper()
        self.insight = insight
        self.onnx_sess = onnx_sess

    def embed(self, img_path: Path) -> np.ndarray | None:
        """
        Load image from *img_path* and return an embedding vector, or None
        if no face is detected.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Could not read image: %s", img_path)
            return None

        if self.model_name == "INSIGHTFACE":
            return self._embed_insightface(img)
        elif self.model_name in ("MOBILEFACENET", "FACENET"):
            return self._embed_onnx(img)

        log.warning("Unknown model '%s'.", self.model_name)
        return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_insightface(self, img: np.ndarray) -> np.ndarray | None:
        faces = self.insight.get(img)
        if not faces:
            return None
        # Return embedding of the largest detected face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding if largest.embedding is not None else None

    def _embed_onnx(self, img: np.ndarray) -> np.ndarray | None:
        if self.onnx_sess is None:
            log.warning("ONNX session not loaded for %s.", self.model_name)
            return None

        # Haar cascade for face detection (same as main.py)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face_img = img[y: y + h, x: x + w]
        if face_img.size == 0:
            return None

        if self.model_name == "MOBILEFACENET":
            blob = cv2.resize(face_img, (112, 112)).astype(np.float32) / 255.0
        else:  # FACENET
            blob = (cv2.resize(face_img, (160, 160)).astype(np.float32) - 127.5) / 128.0

        blob = np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)
        input_name = self.onnx_sess.get_inputs()[0].name
        return self.onnx_sess.run(None, {input_name: blob})[0][0]


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_offline_eval(
    pairs: list[tuple[Path, Path, int]],
    extractors: dict[str, EmbeddingExtractor],
    include_spoof_eval: bool = False,
    attack_type: str = "print_flat",
) -> dict[str, dict]:
    """
    Run all extractors against *pairs* and return per-model metric dicts.

    Parameters
    ----------
    pairs              : list of (img1_path, img2_path, label)
    extractors         : {model_name: EmbeddingExtractor}
    include_spoof_eval : if True, also evaluate anti-spoofing on genuine faces
    attack_type        : spoof attack to apply when include_spoof_eval is True

    Returns
    -------
    { model_name: { "biometric": {...}, "spoofing": {...} } }
    """
    evaluators: dict[str, BiometricEvaluator] = {
        name: BiometricEvaluator() for name in extractors
    }
    spoof_scores: dict[str, list[float]] = {name: [] for name in extractors}
    spoof_labels: dict[str, list[int]]   = {name: [] for name in extractors}
    spoof_sim = SpoofSimulator()

    total = len(pairs)
    log.info("Starting offline evaluation: %d pairs × %d models.", total, len(extractors))

    for idx, (p1, p2, label) in enumerate(pairs, 1):
        if idx % 100 == 0:
            log.info("  Progress: %d / %d", idx, total)

        for model_name, extractor in extractors.items():
            emb1 = extractor.embed(p1)
            emb2 = extractor.embed(p2)

            if emb1 is None or emb2 is None:
                continue    # no face detected — skip pair for this model

            score = cosine_similarity(emb1, emb2)
            evaluators[model_name].add_pair(score, is_genuine=(label == 1))

            # Anti-spoofing: apply attack to genuine faces and test if
            # the embedding drifts (a robust model should be affected less)
            if include_spoof_eval and label == 1:
                img1 = cv2.imread(str(p1))
                if img1 is not None:
                    spoofed = spoof_sim.apply(img1, attack_type)      # type: ignore[arg-type]
                    cv2.imwrite("/tmp/_spoof_tmp.jpg", spoofed)
                    emb_spoof = extractor.embed(Path("/tmp/_spoof_tmp.jpg"))
                    if emb_spoof is not None:
                        spoof_score = cosine_similarity(emb1, emb_spoof)
                        # High score = model fooled by spoof; low = robust
                        spoof_scores[model_name].append(spoof_score)
                        spoof_labels[model_name].append(0)   # 0 = attack
                        # Also add bona-fide pair so APCER/BPCER are meaningful
                        spoof_scores[model_name].append(cosine_similarity(emb1, emb2))
                        spoof_labels[model_name].append(1)   # 1 = live

    results: dict[str, dict] = {}
    for model_name, evaluator in evaluators.items():
        bio = evaluator.compute()
        spoof: dict = {}
        if include_spoof_eval and spoof_scores[model_name]:
            spoof = compute_anti_spoofing_metrics(
                spoof_scores[model_name], spoof_labels[model_name]
            )
        results[model_name] = {"biometric": bio, "spoofing": spoof}

    return results


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_summary(results: dict, path: Path) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("JSON summary → %s", path)


def write_latex_table(results: dict, path: Path) -> None:
    """
    Emit a booktabs-compatible LaTeX table for direct thesis inclusion.
    Requires \\usepackage{booktabs} in your LaTeX preamble.
    """
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Face Recognition Model Benchmark Results}",
        r"\label{tab:benchmark_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & AUC $\uparrow$ & EER (\%) $\downarrow$ & FAR (\%) $\downarrow$ "
        r"& FRR (\%) $\downarrow$ & Pairs \\",
        r"\midrule",
    ]

    for model, data in results.items():
        bio = data.get("biometric", {})
        if not bio:
            continue
        lines.append(
            f"{model} & "
            f"{bio.get('AUC', 0):.4f} & "
            f"{bio.get('EER', 0)*100:.2f} & "
            f"{bio.get('FAR', 0)*100:.2f} & "
            f"{bio.get('FRR', 0)*100:.2f} & "
            f"{bio.get('n_genuine', 0) + bio.get('n_impostor', 0)} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("LaTeX table → %s", path)


def write_roc_plot(results: dict, path: Path) -> None:
    """
    Save an ROC curve comparison plot using matplotlib.
    If matplotlib is not installed, logs a warning and skips.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — ROC plot skipped. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    colors = {"INSIGHTFACE": "#2a78d6", "MOBILEFACENET": "#1baf7a", "FACENET": "#eda100"}

    for model, data in results.items():
        bio = data.get("biometric", {})
        fpr = bio.get("fpr")
        tpr = bio.get("tpr")
        auc_val = bio.get("AUC", 0)
        eer_val = bio.get("EER", 0)
        if fpr and tpr:
            ax.plot(fpr, tpr,
                    label=f"{model}  AUC={auc_val:.3f}  EER={eer_val*100:.1f}%",
                    color=colors.get(model, "gray"),
                    linewidth=1.8)

    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 − FRR)")
    ax.set_title("ROC Curve — Face Recognition Model Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close()
    log.info("ROC plot → %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline face recognition benchmark evaluation."
    )
    parser.add_argument("--dataset", choices=["lfw", "custom"], default="lfw")
    parser.add_argument("--data_dir", default="datasets/lfw",
                        help="Root directory of LFW images.")
    parser.add_argument("--pairs_file", default="pairs.txt",
                        help="LFW pairs.txt filename (inside --data_dir).")
    parser.add_argument("--custom_pairs", default="pairs_custom.csv",
                        help="Path to custom pairs CSV (when --dataset custom).")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs (useful for quick tests).")
    parser.add_argument("--spoof", action="store_true",
                        help="Also run anti-spoofing evaluation.")
    parser.add_argument("--attack", default="print_flat",
                        choices=["print_flat", "print_wrapped", "replay", "mask_3d"],
                        help="Spoof attack type for anti-spoofing eval.")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory for JSON, LaTeX, and ROC plot.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    if args.dataset == "lfw":
        pairs = load_lfw_pairs(args.data_dir, args.pairs_file, args.max_pairs)
    else:
        pairs = load_custom_pairs(args.custom_pairs)
        if args.max_pairs:
            pairs = pairs[: args.max_pairs]

    if not pairs:
        log.error("No valid pairs loaded. Check your dataset path.")
        return

    # Build extractors — lazy import to avoid loading heavy models unless needed
    extractors: dict[str, EmbeddingExtractor] = {}

    try:
        from insightface.app import FaceAnalysis
        insight = FaceAnalysis(name="buffalo_s")
        insight.prepare(ctx_id=-1, det_size=(640, 640))
        extractors["INSIGHTFACE"] = EmbeddingExtractor("INSIGHTFACE", insight=insight)
        log.info("InsightFace loaded.")
    except Exception as exc:
        log.warning("InsightFace not available: %s", exc)

    try:
        import onnxruntime as ort
        mb_sess = ort.InferenceSession("mobilefacenet.onnx", providers=["CPUExecutionProvider"])
        extractors["MOBILEFACENET"] = EmbeddingExtractor("MOBILEFACENET", onnx_sess=mb_sess)
        log.info("MobileFaceNet ONNX loaded.")
    except Exception as exc:
        log.warning("MobileFaceNet not available: %s", exc)

    try:
        import onnxruntime as ort
        fn_sess = ort.InferenceSession("facenet.onnx", providers=["CPUExecutionProvider"])
        extractors["FACENET"] = EmbeddingExtractor("FACENET", onnx_sess=fn_sess)
        log.info("FaceNet ONNX loaded.")
    except Exception as exc:
        log.warning("FaceNet not available: %s", exc)

    if not extractors:
        log.error("No models loaded. Cannot run evaluation.")
        return

    # Run evaluation
    results = run_offline_eval(
        pairs, extractors,
        include_spoof_eval=args.spoof,
        attack_type=args.attack,
    )

    # Write outputs
    write_json_summary(results, out_dir / "benchmark_summary.json")
    write_latex_table(results, out_dir / "benchmark_table.tex")
    write_roc_plot(results, out_dir / "roc_curve.png")

    log.info("Evaluation complete. Outputs in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
