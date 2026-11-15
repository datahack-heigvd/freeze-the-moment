import cv2
import numpy as np


def paint_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 165)) & (s > 55) & (v > 150)
    blue = (h > 100) & (h < 135) & (s > 35) & (s < 150) & (v > 120)
    k = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, k)
    blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, k)
    return red, blue


def ridge_masks(img, thr_r=14, thr_b=14):
    """Thin red / blue paint via local colour contrast (works for faint rings and lines)."""
    f = img.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    redness = R - 0.5 * (G + B)
    blueness = B - 0.5 * (R + G)
    out = []
    for c, thr in ((redness, thr_r), (blueness, thr_b)):
        d = c - cv2.GaussianBlur(c, (0, 0), 9)
        m = (d > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        out.append(m)
    return out
