import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from overlay import draw, project

img = cv2.imread("cache/faceoff_920.png")
# image px -> world, hand-read landmarks (1920x1080)
pairs = [
    ((493, 243), (-7.14, 15)),
    ((953, 243), (0, 15)),
    ((1408, 243), (7.14, 15)),
    ((578, 634), (-4.5, 0)),
    ((1338, 634), (4.5, 0)),
    ((958, 478), (0, 4.5)),
    ((958, 790), (0, -4.5)),
    ((530, 414), (-5.64, 7)),
    ((1372, 409), (5.64, 7)),
    ((345, 917), (-5.64, -7)),
    ((1568, 908), (5.64, -7)),
]
img_pts = np.array([p for p, _ in pairs], np.float64)
world = np.array([w for _, w in pairs], np.float64)
H, mask = cv2.findHomography(world, img_pts, cv2.RANSAC, 6.0)
print(H, mask.ravel())
np.save("cache/H_faceoff_init.npy", H)
res = np.linalg.norm(project(H, world) - img_pts, axis=1)
print("residual px", np.round(res, 1))
cv2.imwrite("cache/overlay_init.jpg", draw(img, H))
