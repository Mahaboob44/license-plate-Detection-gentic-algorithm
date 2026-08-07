"""
Image acquisition & preprocessing
==================================
Mirrors Section 4.1 / 4.2 of the project report:
    - Grayscale conversion   : Gray = 0.2989R + 0.5870G + 0.1140B
    - Canny edge detection   : E = Canny(I_gray)
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the same weighting the report
    specifies: Gray = 0.2989 R + 0.5870 G + 0.1140 B
    """
    b, g, r = cv2.split(image_bgr.astype(np.float64))
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light Gaussian blur to reduce noise before edge detection."""
    return cv2.GaussianBlur(gray, (5, 5), 0)


def canny_edges(gray: np.ndarray, low: int = 80, high: int = 200) -> np.ndarray:
    """Apply Canny edge detection to highlight high-frequency regions."""
    smoothed = denoise(gray)
    return cv2.Canny(smoothed, low, high)


def preprocess(image_bgr: np.ndarray):
    """Convenience wrapper returning (gray, edges) for a BGR image."""
    gray = to_grayscale(image_bgr)
    edges = canny_edges(gray)
    return gray, edges
