import re
import cv2
import numpy as np

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


_TESS_CONFIG = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR accuracy
    h, w = gray.shape
    if h < 40:
        scale = max(2, 40 // h)
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def detect_jersey_number(frame: np.ndarray, bbox: tuple) -> int | None:
    """
    Attempt to OCR the jersey number from a player bounding box.
    Returns the detected integer number, or None if detection fails or
    pytesseract is not installed.
    """
    if not _TESSERACT_AVAILABLE:
        return None

    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # Focus on the upper-torso region where numbers typically appear
    top = y1 + h // 4
    bot = y1 + (3 * h) // 5
    crop = frame[top:bot, x1:x2]

    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    processed = _preprocess_crop(crop)

    try:
        raw = pytesseract.image_to_string(processed, config=_TESS_CONFIG).strip()
    except Exception:
        return None

    digits = re.sub(r"\D", "", raw)
    if digits and 1 <= int(digits) <= 99:
        return int(digits)
    return None
