"""
Neural-Network-based OCR engine
================================
Mirrors Section 3.4 / 4.5 of the project report — recognizes the
alphanumeric characters on the cropped license plate region.

Uses EasyOCR, a deep-learning OCR engine (CRAFT text detector +
CRNN/attention recognizer) as a practical, pip-installable stand-in for the
"Neural Network OCR" module described in the MATLAB prototype. Swap
`recognize()` for a custom-trained CNN+CTC model if you want to reproduce
the project exactly.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

_READER = None  # lazy-loaded singleton


def _get_reader():
    global _READER
    if _READER is None:
        import easyocr  # imported lazily so the GA module has no heavy deps
        _READER = easyocr.Reader(["en"], gpu=False)
    return _READER


def clean_plate_text(raw_text: str) -> str:
    """Keep only A-Z / 0-9, uppercase everything (matches Indian plate format)."""
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())


def recognize(plate_region_bgr: np.ndarray) -> Optional[str]:
    """
    Run NN-based OCR on a cropped license-plate image and return the
    cleaned alphanumeric string, or None if nothing was recognized.
    """
    if plate_region_bgr is None or plate_region_bgr.size == 0:
        return None

    reader = _get_reader()
    results = reader.readtext(plate_region_bgr)
    if not results:
        return None

    # Concatenate all detected text fragments (some plates split into
    # state-code / number blocks), sorted left-to-right by bounding box.
    results.sort(key=lambda r: r[0][0][0])
    combined = "".join(text for _, text, _ in results)
    cleaned = clean_plate_text(combined)
    return cleaned or None
