import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
# security.py
"""
Anti-spoofing evaluation module.
Implements: liveness heuristics + spoof attack simulator + ISO/IEC 30107-3 metrics
"""

class LivenessDetector:
    """
    Passive liveness using texture analysis (LBP) + blink detection.
    For a thesis: reference MN3 or Silent-Face as the deep-learning baseline.
    """
    def __init__(self, method="lbp"):
        self.method = method  # "lbp" | "deep" | "hybrid"

    def score(self, face_img: np.ndarray) -> tuple[float, bool]:
        """Returns (liveness_score 0–1, is_live)."""
        if self.method == "lbp":
            return self._lbp_score(face_img)
        ...

    def _lbp_score(self, img):
        # Local Binary Patterns — print attacks have lower texture variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = float(np.var(gray) / 255.0)          # simplified
        return score, score > 0.12


class SpoofSimulator:
    """
    Generates spoofed face images for benchmark evaluation.
    Attack types per ISO/IEC 30107-3:
      - print_flat    : printed photo (blur + desaturate)
      - print_wrapped : curved print (barrel distortion)
      - replay        : screen glare simulation (specular highlight)
      - mask_3d       : depth-map flattening
    """
    def apply(self, face_img: np.ndarray, attack: str) -> np.ndarray:
        if attack == "print_flat":
            return cv2.GaussianBlur(
                cv2.cvtColor(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR), (3, 3), 0)
        if attack == "replay":
            overlay = face_img.copy()
            cv2.circle(overlay, (face_img.shape[1]//2, face_img.shape[0]//3),
                       30, (255,255,255), -1)
            return cv2.addWeighted(face_img, 0.85, overlay, 0.15, 0)
        # ... add print_wrapped, mask_3d
        return face_img


def compute_anti_spoofing_metrics(scores, labels):
    """
    ISO/IEC 30107-3 metrics:
    APCER: Attack Presentation Classification Error Rate  (spoofs accepted as live)
    BPCER: Bona-fide Presentation Classification Error Rate (live rejected as spoof)
    ACER : Average Classification Error Rate = (APCER + BPCER) / 2
    """
    scores, labels = np.array(scores), np.array(labels)
    threshold = 0.5

    live_mask  = labels == 1
    spoof_mask = labels == 0

    BPCER = np.mean(scores[live_mask]  < threshold)   # live classified as spoof
    APCER = np.mean(scores[spoof_mask] >= threshold)   # spoof classified as live
    ACER  = (APCER + BPCER) / 2.0

    return {"APCER": APCER, "BPCER": BPCER, "ACER": ACER}