"""
Controls:
  [S]  Freeze & save snapshot
  [C]  Clear & unfreeze
  [R]  Reset all metrics
  [Q]  Quit
"""

import cv2
import time
import numpy as np
import psutil
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine


# بارگذاری مدل‌ها

print("=" * 60)
print("  FACE RECOGNITION BENCHMARK")
print("  InsightFace (ArcFace) vs Haar Cascade")
print("=" * 60)

print("\n[INFO] Loading InsightFace Buffalo_L ...")
insight = FaceAnalysis(name='buffalo_l', root='share')
insight.prepare(ctx_id=0, det_size=(320, 320))

print("[INFO] Loading Haar Cascade ...")
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("[INFO] All models ready.\n")

# پایگاه داده embedding

try:
    raw = np.load("fatemeh_db.npy")
    DATABASE = raw.reshape(-1, raw.shape[-1]) if raw.ndim > 1 else raw.reshape(1, -1)
    print(f"[INFO] Database: {len(DATABASE)} embedding(s) loaded.")
    DB_ENABLED = True
except Exception:
    DATABASE   = np.empty((0, 512))
    DB_ENABLED = False
    print("[INFO] No database — identity recognition disabled.")

ID_THRESHOLD = 0.37

# دوربین

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
if not cap.isOpened():
    raise RuntimeError("Camera not found!")
psutil.cpu_percent(interval=None) 

# ثابت‌های نمایش

CAM_W, CAM_H = 640, 480
PANEL_W       = 380
WIN_W         = CAM_W + PANEL_W
WIN_H         = CAM_H

# رنگ‌ها
C_BG        = (18,  18,  28)
C_CARD      = (28,  28,  42)
C_BORDER    = (60,  60,  80)
C_GOLD      = (30,  210, 255)  
C_GREEN     = (80,  220, 80)
C_ORANGE    = (60,  165, 255)   
C_WHITE     = (230, 230, 230)
C_DIM       = (130, 130, 150)
C_RED       = (70,  70,  220)
C_CYAN      = (220, 220,  50)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
AA          = cv2.LINE_AA

# ثابت‌های بنچمارک

INSIGHT_SKIP = 3 

freeze_mode   = False
results       = []
show_msg      = 0
msg_text      = ""
frame_idx     = 0

# آمار تجمعی برای متریک‌های آکادمیک
stats = {
    "ins":  {"times": [], "detected": 0, "total": 0, "fp": 0},
    "haar": {"times": [], "detected": 0, "total": 0, "fp": 0},
}

# آخرین داده هر مدل
EMPTY_DATA = {
    "ms": 0.0, "faces": 0, "fps": 0.0, "cpu": 0.0,
    "identity": "—", "match": 0.0,
    "det_rate": 0.0, "avg_ms": 0.0,
}
live    = {"ins": EMPTY_DATA.copy(), "haar": EMPTY_DATA.copy()}
frozen  = {"ins": EMPTY_DATA.copy(), "haar": EMPTY_DATA.copy()}

# FPS sliding window
_fps_buf: list[float] = []
_last_ins_frame = None   # آخرین فریم annotate‌شده InsightFace

print("Controls:  [S] Freeze & Save   [C] Clear   [R] Reset metrics   [Q] Quit\n")


# توابع کمکی

def get_identity(emb: np.ndarray) -> tuple[str, float]:
    if not DB_ENABLED or DATABASE.shape[0] == 0:
        return "No DB", 0.0
    dists   = [cosine(emb, db) for db in DATABASE]
    min_d   = float(min(dists))
    pct     = float(np.clip((1.0 - min_d) * 100.0, 0, 100))
    name    = "Fatemeh" if min_d < ID_THRESHOLD else "Unknown"
    return name, round(pct, 1)


def fps_tick() -> float:
    now = time.perf_counter()
    _fps_buf.append(now)
    if len(_fps_buf) > 20:
        _fps_buf.pop(0)
    if len(_fps_buf) < 2:
        return 0.0
    return round((len(_fps_buf) - 1) / (_fps_buf[-1] - _fps_buf[0]), 1)


def det_rate(key: str) -> float:
    s = stats[key]
    return round(s["detected"] / s["total"] * 100, 1) if s["total"] > 0 else 0.0


def avg_ms(key: str) -> float:
    t = stats[key]["times"]
    return round(float(np.mean(t)), 1) if t else 0.0


def run_insight(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """InsightFace inference — annotate و داده برمی‌گردونه."""
    t0    = time.perf_counter()
    faces = insight.get(frame)
    ms    = (time.perf_counter() - t0) * 1000.0

    out   = frame.copy()
    identity, match = "No face", 0.0

    stats["ins"]["total"]    += 1
    stats["ins"]["times"].append(ms)
    if len(faces) > 0:
        stats["ins"]["detected"] += 1

    for face in faces:
        identity, match = get_identity(face.embedding)
        b = face.bbox.astype(int)
        # رنگ بر اساس شناسایی
        col = C_GREEN if (identity not in ("Unknown", "No DB", "No face")) else C_ORANGE
        # کادر تمیز با گوشه‌های L-شکل
        _draw_corner_box(out, b[0], b[1], b[2], b[3], col)
        # برچسب
        label = f"{identity}  {match:.0f}%" if DB_ENABLED else "Face detected"
        _label_bg(out, label, b[0], b[1] - 4, col)

    cpu = psutil.cpu_percent(interval=None)
    return out, {
        "ms": round(ms, 1), "faces": len(faces), "fps": 0.0, "cpu": cpu,
        "identity": identity, "match": match,
        "det_rate": det_rate("ins"), "avg_ms": avg_ms("ins"),
    }


def run_haar(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    t0   = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)         
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )
    ms   = (time.perf_counter() - t0) * 1000.0

    out  = frame.copy()
    identity = "No face"

    stats["haar"]["total"]    += 1
    stats["haar"]["times"].append(ms)
    if len(faces) > 0:
        stats["haar"]["detected"] += 1
        identity = "Face"

    for (x, y, w, h) in faces:
        col = (200, 140, 0)   
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        _label_bg(out, "Face", x, y - 4, col)

    cpu = psutil.cpu_percent(interval=None)
    return out, {
        "ms": round(ms, 1), "faces": len(faces), "fps": 0.0, "cpu": cpu,
        "identity": identity, "match": 0.0,
        "det_rate": det_rate("haar"), "avg_ms": avg_ms("haar"),
    }


#   UI

def _draw_corner_box(img, x1, y1, x2, y2, color, length=20, thick=2):
    """کادر با گوشه‌های L-شکل — حرفه‌ای‌تر از مستطیل ساده."""
    # گوشه بالا-چپ
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thick, AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thick, AA)
    # گوشه بالا-راست
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thick, AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thick, AA)
    # گوشه پایین-چپ
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thick, AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thick, AA)
    # گوشه پایین-راست
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thick, AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thick, AA)


def _label_bg(img, text, x, y, color):
    """برچسب با پس‌زمینه تیره برای خوانایی."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    y_base = max(y, th + 4)
    cv2.rectangle(img, (x, y_base - th - 4), (x + tw + 6, y_base + 2),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y_base - 2), FONT, 0.45, color, 1, AA)


def _txt(panel, text, x, y, color=C_WHITE, scale=0.42, thick=1):
    cv2.putText(panel, str(text), (x, y), FONT, scale, color, thick, AA)


def _hline(panel, y, x1=10, x2=None):
    cv2.line(panel, (x1, y), ((x2 or PANEL_W - 10), y), C_BORDER, 1)


def _card(panel, x, y, w, h):
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_CARD, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_BORDER, 1)


def draw_bar(panel, value, max_val, x, y, w=160, h=7, color=C_GREEN):
    ratio = float(np.clip(value / (max_val + 1e-9), 0, 1))
    cv2.rectangle(panel, (x, y), (x + w, y + h), C_BORDER, -1)
    cv2.rectangle(panel, (x, y), (x + int(w * ratio), y + h), color, -1)

def build_panel(d_ins: dict, d_haar: dict,
                frozen: bool, n_saved: int, msg: str) -> np.ndarray:

    panel = np.full((WIN_H, PANEL_W, 3), C_BG, dtype=np.uint8)

    # ───── محاسبه انحراف معیار داخل تابع ─────
    std_ins  = float(np.std(d_ins.get("times", [0]))) if "times" in d_ins else 0.0
    std_haar = float(np.std(d_haar.get("times", [0]))) if "times" in d_haar else 0.0

    # ───── سربرگ ─────
    _card(panel, 6, 10, PANEL_W - 12, 30)
    _txt(panel, "BENCHMARK COMPARISON", 18, 28, C_GOLD, 0.52)

    # ───── ستون‌ها ─────
    _card(panel, 6, 46, PANEL_W - 12, 22)
    _txt(panel, "Metric", 12, 62, C_DIM, 0.40)
    _txt(panel, "InsightFace", 150, 62, C_GREEN, 0.40)
    _txt(panel, "Haar", 290, 62, C_ORANGE, 0.40)

    # ───── داده‌ها (FIX مهم: همه rows دقیقاً 3 مقدار دارند) ─────
    rows = [
        ("Speed (ms)",   d_ins.get("ms", 0),         d_haar.get("ms", 0)),
        ("Avg Speed(ms)",d_ins.get("avg_ms", 0),     d_haar.get("avg_ms", 0)),
        ("Std Dev(ms)",  std_ins,                    std_haar),
        ("Det. Rate(%)", d_ins.get("det_rate", 0),   d_haar.get("det_rate", 0)),
        ("Faces found",  d_ins.get("faces", 0),      d_haar.get("faces", 0)),
        ("FPS",          d_ins.get("fps", 0),        d_haar.get("fps", 0)),
        ("CPU (%)",      d_ins.get("cpu", 0),        d_haar.get("cpu", 0)),
        ("Identity",     str(d_ins.get("identity", "N/A"))[:10], "N/A"),
        ("Match (%)",    d_ins.get("match", 0) if DB_ENABLED else "No DB", "—"),
    ]

    y = 88

    for i, row in enumerate(rows):

        if len(row) != 3:
            continue

        label, v1, v2 = row

        bg = C_CARD if i % 2 == 0 else C_BG
        cv2.rectangle(panel, (6, y - 14), (PANEL_W - 6, y + 6), bg, -1)

        _txt(panel, label, 12, y, C_DIM, 0.38)
        _txt(panel, str(v1), 150, y, C_GREEN, 0.40)
        _txt(panel, str(v2), 290, y, C_ORANGE, 0.40)

        y += 24

    _hline(panel, y)
    y += 12

    _txt(panel, "Speed bars (lower = better):", 12, y, C_DIM, 0.36)
    y += 16

    max_ms = max(float(d_ins.get("ms", 1)), float(d_haar.get("ms", 1)), 1.0)

    _txt(panel, "Ins:", 12, y, C_GREEN, 0.36)
    draw_bar(panel,
             float(d_ins.get("ms", 0)),
             max_ms,
             45, y - 8, 180, 8,
             C_GREEN if d_ins.get("ms", 0) <= d_haar.get("ms", 0) else C_RED)

    _txt(panel, f"{d_ins.get('ms', 0):.0f}ms", 232, y, C_WHITE, 0.35)
    y += 20

    return panel

def make_split_frame(f_ins: np.ndarray, f_haar: np.ndarray) -> np.ndarray:
    half_h = CAM_H // 2
    # crop مرکزی به نسبت صحیح
    def fit(img, tw, th):
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    top    = fit(f_ins,  CAM_W, half_h)
    bottom = fit(f_haar, CAM_W, half_h)

    # برچسب مدل
    _label_bg(top,    "InsightFace (ArcFace)", 6, 18, C_GREEN)
    _label_bg(bottom, "Haar Cascade (Classic)", 6, 18, C_ORANGE)

    # خط جداکننده
    combined = np.vstack([top, bottom])
    cv2.line(combined, (0, half_h), (CAM_W, half_h), C_BORDER, 2)
    return combined


def save_results():
    if not results:
        return "Nothing to save."
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FACE RECOGNITION BENCHMARK RESULTS\n")
        f.write(f"Date: {time.ctime()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'#':<4} {'InsightFace ms':>16} {'Identity':>12} "
                f"{'Match%':>8} {'Haar ms':>10} {'Haar Faces':>12}\n")
        f.write("-" * 60 + "\n")
        for i, r in enumerate(results, 1):
            f.write(
                f"{i:<4} {r['ins_ms']:>16.0f} {r['ins_id']:>12} "
                f"{r['ins_match']:>8.1f} {r['haar_ms']:>10.0f} {r['haar_faces']:>12}\n"
        
            )
    return f"Saved {len(results)} result(s) → benchmark_results.txt"


# حلقه اصلی

last_ins_annotated = None     

while True:
    ret, raw = cap.read()
    if not ret:
        continue

    frame_idx += 1
    fps_val = fps_tick()

    if not freeze_mode:
        f_haar, d_haar = run_haar(raw)
        d_haar["fps"]  = fps_val

        # InsightFace هر INSIGHT_SKIP فریم — برای حفظ FPS
        if frame_idx % INSIGHT_SKIP == 0 or last_ins_annotated is None:
            f_ins, d_ins       = run_insight(raw)
            d_ins["fps"]       = fps_val
            last_ins_annotated = (f_ins, d_ins)
        else:
            f_ins, d_ins = last_ins_annotated
            d_ins = d_ins.copy()
            d_ins["fps"]   = fps_val
            d_ins["cpu"]   = psutil.cpu_percent(interval=None)

        live["ins"]  = d_ins
        live["haar"] = d_haar

    else:
        f_haar, _ = run_haar(raw)
        if frame_idx % INSIGHT_SKIP == 0 or last_ins_annotated is None:
            f_ins, _ = run_insight(raw)
            last_ins_annotated = (f_ins, _)
        else:
            f_ins, _ = last_ins_annotated

    show = frozen if freeze_mode else live

    split  = make_split_frame(f_ins, f_haar)
    panel  = build_panel(show["ins"], show["haar"],
                         freeze_mode, len(results),
                         msg_text if show_msg > 0 else "")

    if show_msg > 0:
        show_msg -= 1
    else:
        msg_text = ""

    window = np.hstack([split, panel])
    cv2.imshow("Face Recognition Benchmark", window)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        freeze_mode    = True
        frozen["ins"]  = live["ins"].copy()
        frozen["haar"] = live["haar"].copy()
        results.append({
            "ins_ms":    live["ins"]["ms"],
            "ins_id":    live["ins"]["identity"],
            "ins_match": live["ins"]["match"],
            "haar_ms":   live["haar"]["ms"],
            "haar_faces":live["haar"]["faces"],
        })
        msg = save_results()
        msg_text  = f"Frozen! {msg}"
        show_msg  = 100
        print(f"[FROZEN] Ins:{live['ins']['ms']:.0f}ms  Haar:{live['haar']['ms']:.0f}ms")

    elif key == ord('c'):
        freeze_mode = False
        msg_text    = "Cleared & live again."
        show_msg    = 60
        print("[LIVE]")

    elif key == ord('r'):
        stats["ins"]  = {"times": [], "detected": 0, "total": 0, "fp": 0}
        stats["haar"] = {"times": [], "detected": 0, "total": 0, "fp": 0}
        results.clear()
        msg_text  = "Metrics reset."
        show_msg  = 60
        print("[RESET]")

    elif key == ord('q'):
        save_results()
        break

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] Results saved to benchmark_results.txt")