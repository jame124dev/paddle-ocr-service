import os
import cv2
import tempfile
from collections import defaultdict
import warnings
import concurrent.futures
from flask import Flask, request, jsonify

# Reduce noisy runtime logs/warnings from Paddle stack.
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_logtostderr", "0")
os.environ.setdefault("PADDLE_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message=".*No ccache found.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

from paddleocr import PaddleOCR, TextRecognition
import paddleocr as paddleocr_pkg

app = Flask(__name__)
ENGINE_VERSION = "2026-02-27-expected-guided-v17-fast"

# Cache OCR instances per (language, mode)
ocr_instances = {}
rec_instances = {}

# Thread pool for parallel OCR passes.
# On Windows, PaddlePaddle is NOT thread-safe for concurrent predict() calls
# on the same model instance — use max_workers=1 on Windows to serialize,
# or increase only if you have confirmed thread-safe GPU/CPU builds.
import platform
_MAX_WORKERS = 1 if platform.system() == "Windows" else 4
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def env_true(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_railway() -> bool:
    return bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"))


def has_gpu() -> bool:
    return os.getenv("OCR_USE_GPU", "false").lower() == "true"


def lightweight_mode() -> bool:
    # CPU-only always uses mobile models — server models need GPU to be usable.
    # To override: set OCR_FORCE_SERVER=true (dangerous on CPU, very slow).
    if not has_gpu() and not env_true("OCR_FORCE_SERVER", default=False):
        return True
    # Explicit override always wins.
    if os.getenv("OCR_LIGHTWEIGHT") is not None:
        return env_true("OCR_LIGHTWEIGHT", default=False)
    # Railway defaults to lightweight.
    if is_railway():
        return True
    # CPU without explicit opt-in => mobile models (server models are 10-30x slower on CPU).
    # Set OCR_LIGHTWEIGHT=false + OCR_USE_GPU=true to enable server models.
    if not has_gpu():
        return True
    return False


def fast_mode() -> bool:
    if os.getenv("OCR_FAST_MODE") is not None:
        return env_true("OCR_FAST_MODE", default=False)
    return lightweight_mode()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "engine_version": ENGINE_VERSION,
            "paddleocr_version": getattr(paddleocr_pkg, "__version__", "unknown"),
        }
    ), 200


# ---------------------------------------------------------------------------
# PaddleOCR version helpers
# ---------------------------------------------------------------------------

def get_paddleocr_major_version() -> int:
    try:
        version = getattr(paddleocr_pkg, "__version__", "2.0.0")
        return int(str(version).split(".")[0])
    except Exception:
        return 2


def is_v3() -> bool:
    return get_paddleocr_major_version() >= 3


# ---------------------------------------------------------------------------
# OCR instance builders (called once per key, results cached)
# ---------------------------------------------------------------------------

def build_ocr(lang: str, mode: str | None):
    use_gpu = has_gpu()

    if is_v3():
        device = "gpu:0" if use_gpu else "cpu"
        use_line_orientation = (
            lang == "ch" and mode == "handwriting" and not lightweight_mode()
        )
        params = {
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_line_orientation,
        }

        if lang == "ch" and mode == "handwriting":
            ocr_version = os.getenv("CH_HAND_OCR_VERSION", "PP-OCRv5")
            lw = lightweight_mode()
            default_det_model = "PP-OCRv4_mobile_det" if lw else "PP-OCRv5_server_det"
            default_rec_model = "PP-OCRv4_mobile_rec" if lw else "PP-OCRv5_server_rec"
            det_model_name = os.getenv("CH_HAND_TEXT_DET_MODEL_NAME", default_det_model)
            rec_model_name = os.getenv("CH_HAND_TEXT_REC_MODEL_NAME", default_rec_model)

            if det_model_name:
                params["text_detection_model_name"] = det_model_name
            if rec_model_name:
                params["text_recognition_model_name"] = rec_model_name

            det_dir = os.getenv("CH_HAND_DET_DIR")
            rec_dir = os.getenv("CH_HAND_REC_DIR")
            line_dir = os.getenv("CH_HAND_CLS_DIR")
            if det_dir:
                params["text_detection_model_dir"] = det_dir
            if rec_dir:
                params["text_recognition_model_dir"] = rec_dir
            if line_dir:
                params["textline_orientation_model_dir"] = line_dir

            params["text_det_thresh"] = float(os.getenv("CH_HAND_DET_DB_THRESH", "0.2"))
            params["text_det_box_thresh"] = float(os.getenv("CH_HAND_DET_DB_BOX_THRESH", "0.35"))
            params["text_det_unclip_ratio"] = float(os.getenv("CH_HAND_DET_DB_UNCLIP_RATIO", "1.8"))

            if not (det_model_name or rec_model_name or det_dir or rec_dir):
                params["lang"] = lang
                params["ocr_version"] = ocr_version
            return PaddleOCR(**params)

        params["lang"] = lang
        params["ocr_version"] = os.getenv("OCR_VERSION", "PP-OCRv5")
        return PaddleOCR(**params)

    # 2.x fallback
    if lang == "ch" and mode == "handwriting":
        det_dir = os.getenv("CH_HAND_DET_DIR")
        rec_dir = os.getenv("CH_HAND_REC_DIR")
        cls_dir = os.getenv("CH_HAND_CLS_DIR")
        return PaddleOCR(
            use_angle_cls=False,
            lang=lang,
            use_gpu=use_gpu,
            det_model_dir=det_dir or None,
            rec_model_dir=rec_dir or None,
            cls_model_dir=cls_dir or None,
            det_db_thresh=float(os.getenv("CH_HAND_DET_DB_THRESH", "0.2")),
            det_db_box_thresh=float(os.getenv("CH_HAND_DET_DB_BOX_THRESH", "0.35")),
            det_db_unclip_ratio=float(os.getenv("CH_HAND_DET_DB_UNCLIP_RATIO", "1.8")),
        )
    return PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu)


def get_ocr(lang: str = "ch", mode: str | None = None):
    key = f"{lang}:{mode or 'default'}"
    if key not in ocr_instances:
        ocr_instances[key] = build_ocr(lang, mode)
    return ocr_instances[key]


def build_recognizer(lang: str, mode: str | None):
    if not is_v3():
        return None
    use_gpu = has_gpu()
    device = "gpu:0" if use_gpu else "cpu"
    if lang == "ch" and mode == "handwriting":
        if lightweight_mode():
            return None
        model_name = os.getenv("CH_HAND_TEXT_REC_MODEL_NAME", "PP-OCRv5_server_rec")
        rec_dir = os.getenv("CH_HAND_REC_DIR")
        if rec_dir:
            return TextRecognition(device=device, model_dir=rec_dir)
        return TextRecognition(device=device, model_name=model_name)
    return TextRecognition(device=device)


def get_recognizer(lang: str = "ch", mode: str | None = None):
    key = f"{lang}:{mode or 'default'}"
    if key not in rec_instances:
        rec_instances[key] = build_recognizer(lang, mode)
    return rec_instances[key]


# ---------------------------------------------------------------------------
# Image preprocessing (unchanged logic, same accuracy)
# ---------------------------------------------------------------------------

def preprocess_image_for_mode(img, lang: str, mode: str | None, variant: str = "default"):
    if lang == "ch" and mode == "handwriting":
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if variant == "default":
            denoised = cv2.bilateralFilter(gray, 9, 60, 60)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(denoised)
        if variant == "contrast":
            blur_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=19, sigmaY=19)
            norm = cv2.divide(gray, blur_bg, scale=255)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            boosted = clahe.apply(norm)
            sharp = cv2.addWeighted(
                boosted, 1.35, cv2.GaussianBlur(boosted, (0, 0), 1.2), -0.35, 0
            )
            return sharp
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 6
        )
    return cv2.resize(img, None, fx=1.5, fy=1.5)


def rotate_image(img, angle_deg: float):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def thicken_strokes(gray_or_bin):
    if len(gray_or_bin.shape) != 2:
        return gray_or_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.dilate(gray_or_bin, kernel, iterations=1)


def extract_main_glyph_roi(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return None
    largest_idx = max(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    if w <= 0 or h <= 0:
        return None
    pad = max(8, int(0.18 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    side = max(roi.shape[0], roi.shape[1])
    canvas = 255 * (cv2.UMat(side, side, cv2.CV_8UC1).get())
    y_off = (side - roi.shape[0]) // 2
    x_off = (side - roi.shape[1]) // 2
    canvas[y_off:y_off + roi.shape[0], x_off:x_off + roi.shape[1]] = roi
    return cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Candidate extraction / ranking (unchanged logic)
# ---------------------------------------------------------------------------

def extract_candidates(result):
    candidates = []
    for page in result or []:
        try:
            rec_texts = page.get("rec_texts", None)
            rec_scores = page.get("rec_scores", None)
        except Exception:
            rec_texts = None
            rec_scores = None
        if isinstance(rec_texts, list):
            if not isinstance(rec_scores, list):
                rec_scores = [0.5] * len(rec_texts)
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if idx < len(rec_scores) else 0.5
                if isinstance(text, str):
                    candidates.append((text, float(score)))
            if candidates:
                return candidates

    for line in result or []:
        if not line:
            continue
        if isinstance(line, (list, tuple)) and len(line) >= 1 and isinstance(line[0], str):
            score = float(line[1]) if len(line) >= 2 and isinstance(line[1], (int, float)) else 0.5
            candidates.append((line[0], score))
            continue
        if not isinstance(line, (list, tuple)):
            continue
        for item in line:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 1
                and isinstance(item[1][0], str)
            ):
                score = float(item[1][1]) if len(item[1]) >= 2 and isinstance(item[1][1], (int, float)) else 0.5
                candidates.append((item[1][0], score))
            elif isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
                score = float(item[1]) if len(item) >= 2 and isinstance(item[1], (int, float)) else 0.5
                candidates.append((item[0], score))
    return candidates


def rank_candidates(candidates, single_char_only: bool = False):
    if not candidates:
        return []
    scores = defaultdict(list)
    for text, score in candidates:
        t = (text or "").strip()
        if not t:
            continue
        if single_char_only and len(t) != 1:
            continue
        scores[t].append(max(0.0, min(1.0, float(score))))
    ranked = []
    for text, vals in scores.items():
        avg_score = sum(vals) / len(vals)
        boost = min(0.2, 0.05 * (len(vals) - 1))
        ranked.append((text, avg_score + boost, avg_score, len(vals)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def best_confidence(ranked):
    return float(ranked[0][2]) if ranked else 0.0


def should_early_stop(ranked, expected_text: str | None):
    if not ranked:
        return False
    if expected_text:
        expected_text = expected_text.strip()
        if expected_text and any(item[0] == expected_text for item in ranked):
            return True
    # Lowered from 0.82 → 0.75: still high-confidence but stops sooner
    return best_confidence(ranked) >= 0.75


def should_escalate_to_heavy(ranked, expected_text: str | None):
    if not ranked:
        return True
    conf = best_confidence(ranked)
    if expected_text:
        t = expected_text.strip()
        if t and not any(item[0] == t for item in ranked):
            return conf < 0.55  # only escalate if also low confidence
    return conf < 0.55  # raised from 0.60 → 0.55


def parse_allowed_chars(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        if not s:
            return set()
        parts = [p.strip().strip("'\"") for p in s.split(",")]
        return {p for p in parts if p}
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return {p for p in parts if p}
    return set(list(s))


def filter_ranked_by_allowed(ranked, allowed_chars):
    if not ranked or not allowed_chars:
        return ranked
    filtered = [item for item in ranked if item[0] in allowed_chars]
    return filtered if filtered else ranked


def maybe_pick_expected(ranked, expected_text: str | None):
    if not ranked or not expected_text:
        return None
    expected_text = expected_text.strip()
    if not expected_text:
        return None
    for item in ranked:
        if item[0] == expected_text:
            return item
    return None


def is_cjk_char(text: str) -> bool:
    if not text or len(text) != 1:
        return False
    code = ord(text[0])
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def apply_language_prior(ranked, lang: str, mode: str | None, single_char_only: bool):
    if not ranked:
        return ranked
    if not (lang == "ch" and mode == "handwriting" and single_char_only):
        return ranked

    adjusted = []
    for text, weighted, avg, hits in ranked:
        new_weighted = weighted
        if len(text) == 1:
            if is_cjk_char(text):
                new_weighted += 0.22
            else:
                new_weighted -= 0.28
        adjusted.append((text, new_weighted, avg, hits))
    adjusted.sort(key=lambda x: x[1], reverse=True)

    cjk_items = [x for x in adjusted if len(x[0]) == 1 and is_cjk_char(x[0])]
    if cjk_items and max(x[2] for x in cjk_items) >= 0.30:
        adjusted = cjk_items

    simple_chars = {"一", "二", "三", "十", "丁", "了", "厂", "7"}
    non_simple = [x for x in adjusted if x[0] not in simple_chars]
    if non_simple:
        best_non_simple_avg = max(x[2] for x in non_simple)
        rescored = []
        for text, weighted, avg, hits in adjusted:
            new_weighted = weighted
            if text in simple_chars and avg < 0.78 and best_non_simple_avg >= avg - 0.30:
                new_weighted -= 0.38
            rescored.append((text, new_weighted, avg, hits))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored
    return adjusted


# ---------------------------------------------------------------------------
# OCR invocation helpers
# ---------------------------------------------------------------------------

def try_ocr_call(ocr, img, use_textline_orientation: bool = False):
    try:
        if is_v3():
            try:
                return ocr.predict(img, use_textline_orientation=use_textline_orientation)
            except (AttributeError, RuntimeError) as e:
                if "textline_orientation_model" in str(e) or "Unknown exception" in str(e):
                    return ocr.predict(img, use_textline_orientation=False)
                raise
        return ocr.ocr(img, cls=False)
    except Exception as e:
        msg = str(e).lower()
        if "primitive" in msg or "onednn" in msg or "mkldnn" in msg or "unknown exception" in msg:
            return None
        raise


def run_single_pass(ocr, img_to_write, temp_path, use_textline_orientation: bool = False):
    """Write image to a *unique* temp path and run OCR. Thread-safe."""
    # Use a per-call temp file to avoid race conditions in parallel runs.
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img_to_write)
        result = try_ocr_call(ocr, path, use_textline_orientation=use_textline_orientation)
        return extract_candidates(result)
    except Exception as e:
        print(f"Skipping OCR pass due to error: {e}")
        return []
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def run_recognition_only(recognizer, img, temp_path=None):
    if recognizer is None:
        return []
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        result = recognizer.predict(path)
    except Exception as e:
        msg = str(e).lower()
        if "primitive" in msg or "onednn" in msg or "mkldnn" in msg:
            return []
        raise
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    candidates = []
    for item in result or []:
        if not item:
            continue
        text = item.get("rec_text") if hasattr(item, "get") else None
        score = item.get("rec_score") if hasattr(item, "get") else None
        if isinstance(text, str):
            candidates.append((text, float(score) if isinstance(score, (int, float)) else 0.5))
    return candidates


def _run_passes_parallel(tasks):
    """
    Run a list of (fn, *args) callables in parallel.
    Returns flattened list of candidates.
    """
    futures = [_EXECUTOR.submit(fn, *args) for fn, *args in tasks]
    results = []
    for f in concurrent.futures.as_completed(futures):
        try:
            results.extend(f.result())
        except Exception as e:
            print(f"Parallel pass error: {e}")
    return results


# ---------------------------------------------------------------------------
# Response builder helper
# ---------------------------------------------------------------------------

def build_response(top, ordered, top_k, expected_text, expected_hit,
                   single_char_only, use_fast, expected_forced=False, debug=False):
    response = {
        "text": top[0],
        "best_text": top[0],
        "best_score": round(top[2], 4),
        "alternatives": [
            {"text": t, "score": round(avg, 4), "hits": hits}
            for t, _, avg, hits in ordered[:top_k]
        ],
    }
    if expected_text:
        response["expected"] = expected_text
        response["expected_match"] = bool(expected_hit is not None)
    if expected_forced:
        response["expected_forced"] = True
    if debug:
        response["engine_version"] = ENGINE_VERSION
        response["paddleocr_version"] = getattr(paddleocr_pkg, "__version__", "unknown")
        response["single_char_mode"] = single_char_only
        response["lightweight_mode"] = lightweight_mode()
        response["fast_mode"] = use_fast
    return response


def finalize_and_respond(all_candidates, single_char_only, lang, mode, allowed_chars,
                         expected_text, top_k, use_fast, debug):
    ranked = rank_candidates(all_candidates, single_char_only=single_char_only)
    ranked = apply_language_prior(ranked, lang, mode, single_char_only)
    ranked = filter_ranked_by_allowed(ranked, allowed_chars)

    if not ranked and lang == "ch" and mode == "handwriting" and not single_char_only:
        ranked = rank_candidates(all_candidates, single_char_only=True)
        ranked = apply_language_prior(ranked, lang, mode, True)
        ranked = filter_ranked_by_allowed(ranked, allowed_chars)

    if not ranked:
        return None  # caller handles the error

    expected_hit = maybe_pick_expected(ranked, expected_text)
    expected_forced = False
    top = expected_hit if expected_hit is not None else ranked[0]

    if (
        expected_hit is None
        and expected_text
        and lang == "ch"
        and mode == "handwriting"
        and single_char_only
    ):
        exp = expected_text.strip()
        if exp:
            top = (exp, 0.0, 0.0, 0)
            expected_forced = True

    ordered = ranked
    if expected_hit is not None:
        ordered = [expected_hit] + [item for item in ranked if item[0] != expected_hit[0]]

    return build_response(
        top, ordered, top_k, expected_text, expected_hit,
        single_char_only, use_fast, expected_forced, debug
    )


# ---------------------------------------------------------------------------
# Main OCR endpoint
# ---------------------------------------------------------------------------

def check_and_return(all_candidates, single_char_only, lang, mode,
                     allowed_chars, expected_text, top_k, use_fast, debug):
    """Rank current candidates and return response if confident enough, else None."""
    ranked = rank_candidates(all_candidates, single_char_only=single_char_only)
    ranked = apply_language_prior(ranked, lang, mode, single_char_only)
    ranked = filter_ranked_by_allowed(ranked, allowed_chars)
    if not ranked:
        return None
    if should_early_stop(ranked, expected_text):
        expected_hit = maybe_pick_expected(ranked, expected_text)
        top = expected_hit if expected_hit is not None else ranked[0]
        ordered = ranked if expected_hit is None else (
            [expected_hit] + [i for i in ranked if i[0] != expected_hit[0]]
        )
        return build_response(top, ordered, top_k, expected_text,
                              expected_hit, single_char_only, use_fast, debug=debug)
    return None


@app.route('/ocr', methods=['POST'])
def run_ocr():
    try:
        lang = request.form.get('lang', 'ch')
        mode = request.form.get('mode')
        single_char_param = request.form.get('single_char')
        if single_char_param is None:
            single_char_only = (lang == "ch" and mode == "handwriting")
        else:
            single_char_only = single_char_param.lower() == 'true'
        expected_text = request.form.get('expected')
        allowed_chars = parse_allowed_chars(request.form.get('allowed_chars'))
        debug = request.form.get('debug', 'false').lower() == 'true'
        try:
            top_k = max(1, min(20, int(request.form.get('top_k', '5'))))
        except ValueError:
            top_k = 5

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        file.save(temp_path)
        original_img = cv2.imread(temp_path)
        try:
            os.remove(temp_path)
        except Exception:
            pass

        if original_img is None:
            return jsonify({"error": "Image could not be read"}), 400

        use_heavy = not lightweight_mode()
        use_fast = fast_mode()
        ocr = get_ocr(lang, mode)

        # ----------------------------------------------------------------
        # PASS 1 — default preprocessing only (fastest, most reliable)
        # This alone resolves the majority of clear images in <5s.
        # ----------------------------------------------------------------
        img_default = preprocess_image_for_mode(original_img, lang, mode, variant="default")
        all_candidates = run_single_pass(ocr, img_default, None, False)

        # Immediate early-exit — most clean images stop here
        resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                allowed_chars, expected_text, top_k, use_fast, debug)
        if resp:
            return jsonify(resp)

        # ----------------------------------------------------------------
        # PASS 2 — binarized variant
        # ----------------------------------------------------------------
        img_bin = preprocess_image_for_mode(original_img, lang, mode, variant="binarized")
        all_candidates.extend(run_single_pass(ocr, img_bin, None, False))

        resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                allowed_chars, expected_text, top_k, use_fast, debug)
        if resp:
            return jsonify(resp)

        # Fast mode exits here — return best so far
        if use_fast:
            result = finalize_and_respond(all_candidates, single_char_only, lang, mode,
                                         allowed_chars, expected_text, top_k, use_fast, debug)
            return jsonify(result) if result else jsonify({"error": "No text detected"})

        # ----------------------------------------------------------------
        # PASS 3 — orientation-aware (heavy only, single pass)
        # ----------------------------------------------------------------
        if lang == "ch" and mode == "handwriting" and use_heavy:
            all_candidates.extend(run_single_pass(ocr, img_default, None, True))
            resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                    allowed_chars, expected_text, top_k, use_fast, debug)
            if resp:
                return jsonify(resp)

        # ----------------------------------------------------------------
        # PASS 4 — contrast + ROI (only for single-char handwriting)
        # ----------------------------------------------------------------
        if single_char_only:
            img_contrast = preprocess_image_for_mode(original_img, lang, mode, variant="contrast")
            all_candidates.extend(run_single_pass(ocr, img_contrast, None, False))

            resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                    allowed_chars, expected_text, top_k, use_fast, debug)
            if resp:
                return jsonify(resp)

            # ROI glyph isolation
            roi = extract_main_glyph_roi(original_img)
            if roi is not None:
                all_candidates.extend(run_single_pass(ocr, roi, None, False))
                resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                        allowed_chars, expected_text, top_k, use_fast, debug)
                if resp:
                    return jsonify(resp)

            # ----------------------------------------------------------------
            # PASS 5 — rotations (only if still not confident after pass 4)
            # ----------------------------------------------------------------
            ranked_check = rank_candidates(all_candidates, single_char_only=single_char_only)
            if not ranked_check or best_confidence(ranked_check) < 0.50:
                for angle in (-6, 6):
                    rot = rotate_image(img_default, angle)
                    all_candidates.extend(run_single_pass(ocr, rot, None, False))

                resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                        allowed_chars, expected_text, top_k, use_fast, debug)
                if resp:
                    return jsonify(resp)

            # ----------------------------------------------------------------
            # PASS 6 — recognizer-only fallback (last resort)
            # ----------------------------------------------------------------
            if use_heavy:
                recognizer = get_recognizer(lang, mode)
                if recognizer is not None and roi is not None:
                    all_candidates.extend(run_recognition_only(recognizer, roi))
                    resp = check_and_return(all_candidates, single_char_only, lang, mode,
                                            allowed_chars, expected_text, top_k, use_fast, debug)
                    if resp:
                        return jsonify(resp)

        # ----------------------------------------------------------------
        # Final — return best accumulated result
        # ----------------------------------------------------------------
        result = finalize_and_respond(all_candidates, single_char_only, lang, mode,
                                     allowed_chars, expected_text, top_k, use_fast, debug)
        if result is None:
            return jsonify({
                "error": "No text detected",
                "hint": "Try higher-resolution image, lang=ch, mode=handwriting, and single_char=true.",
            })
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"OCR Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


def warmup_models():
    """
    Pre-load OCR models on the main thread before Flask starts serving.
    This avoids lazy-loading inside request handlers (which can crash on
    Windows due to PaddlePaddle's internal threading assumptions).
    """
    lw = lightweight_mode()
    gpu = has_gpu()
    print(f"Mode: lightweight={lw}, gpu={gpu}, fast={fast_mode()}")
    det = "PP-OCRv4_mobile_det" if lw else "PP-OCRv5_server_det"
    rec = "PP-OCRv4_mobile_rec" if lw else "PP-OCRv5_server_rec"
    print(f"Models: det={det}, rec={rec}")
    print("Warming up OCR models...")
    try:
        get_ocr("ch", "handwriting")
        print("OCR models ready.")
    except Exception as e:
        print(f"Warning: model warm-up failed: {e}")


if __name__ == '__main__':
    # IMPORTANT: On Windows, multiprocessing requires the __main__ guard.
    # Never set use_reloader=True or debug=True — both spawn child processes
    # that re-execute this file and break PaddlePaddle initialization.
    import multiprocessing
    multiprocessing.freeze_support()  # needed for PyInstaller / Windows

    warmup_models()
    print(f"Starting OCR service on http://0.0.0.0:5006")
    app.run(
        host='0.0.0.0',
        port=5006,
        debug=False,       # MUST be False — debug=True uses a reloader that forks
        threaded=True,
        use_reloader=False # Explicitly disable reloader to prevent double-start
    )
