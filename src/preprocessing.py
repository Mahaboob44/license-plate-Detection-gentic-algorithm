import cv2
import numpy as np

def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def canny_edges(gray: np.ndarray) -> np.ndarray:
    """Run Canny edge detection on a grayscale image."""
    return cv2.Canny(gray, 100, 200)
    
def get_plate_candidates(image_bgr: np.ndarray, edges: np.ndarray = None) -> list:
    """
    Locate plate-shaped candidate regions from the edge map produced by
    canny_edges(). Used to seed the GA search space so it doesn't waste
    generations exploring badges, stickers, or other non-plate text.

    Returns a list of (x, y, w, h) boxes, sorted largest-area first.
    """
    if edges is None:
        gray = to_grayscale(image_bgr)
        edges = canny_edges(gray)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = edges.shape[:2]
    candidates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)

        # Indian plates are roughly 4.5:1 to 5.5:1
        if 2.5 < aspect < 6.0 and w > 0.08 * img_w and h > 0.02 * img_h:
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    return candidates[:20]