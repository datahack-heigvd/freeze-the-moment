"""Where equipment covers the MHR rest mesh. A helmet: above the brow in front, below the ears at the sides, down to the nape at the
back; a goalie's mask also down over the face to under the chin (viewer/helmet.js draws its shell over the same outline). A goalie's
leg pads: the front of the legs from the ankle to above the knee; their gloves: the hands (viewer/goalie.js). Used by scene_export.py (the helmet's thickness),
gear_colour.py and player_textures.py (their colours)."""

import numpy as np

EARS, CROWN = (256, 3142), 2810  # MHR vertices
REACH = 0.2  # m from between the ears


def head_frame(rest):
    """Rest-mesh points [..., 3] (m) in the head's frame: x to the left ear, y up through the crown, z forward; origin between the ears."""
    L, R = rest[EARS[0]], rest[EARS[1]]
    o = (L + R) / 2
    x = (L - R) / np.linalg.norm(L - R)
    y = rest[CROWN] - o
    y = y - x * (y @ x)
    y = y / np.linalg.norm(y)
    return lambda p: (p - o) @ np.stack([x, y, np.cross(x, y)]).T


def smoothstep(e0, e1, t):
    u = np.clip((t - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


def rim(a, mask=False):
    """Lowest point of the helmet, or a goalie's mask (m above the ears), at azimuth a (rad, 0 forward)."""
    a = np.abs(a)
    back = smoothstep(2.0, 2.8, a)
    if mask:
        front = smoothstep(1.3, 0.8, a)
        return (-0.07 * (1 - back) - 0.065 * back) * (1 - front) - 0.12 * front
    front = smoothstep(1.1, 0.6, a)
    return (-0.035 * (1 - back) - 0.065 * back) * (1 - front) + 0.035 * front


def covered(rest, points, edge, mask=False):
    """How much each of points [..., 3] (on or near the rest mesh, m) is under the helmet (or mask): 1 above its rim, 0 edge below it."""
    h = head_frame(rest)(points)
    lo = rim(np.arctan2(h[..., 0], h[..., 2]), mask)
    return smoothstep(lo - edge, lo, h[..., 1]) * (np.linalg.norm(h, axis=-1) < REACH)


PAD_LEG, PAD_FRONT = (0.02, 0.68), -0.06  # m: height range on the rest mesh; forward of this (the leg's middle)


def padded(points):
    """Whether each of points [..., 3] (on the rest mesh, m) is under a goalie's leg pad: the front half of a leg, ankle to thigh."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return (np.abs(x) > 0.03) & (y > PAD_LEG[0]) & (y < PAD_LEG[1]) & (z > PAD_FRONT)


HAND = 0.2  # m: a hand, from its fingertips in along the arm (the rest mesh's arms are out)


def gloved(points, rest):
    """Whether each of points [..., 3] (on the rest mesh, m) is a hand, under a goalie's blocker or catching glove."""
    return np.abs(points[..., 0]) > np.abs(rest[:, 0]).max() - HAND
