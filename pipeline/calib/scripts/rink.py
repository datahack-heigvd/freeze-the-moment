"""Rink model in metres: origin at centre ice, x along the length, y across."""

import numpy as np

LENGTH, WIDTH, CORNER = 59.472, 30.39, 8.637
BLUE_LINE = 7.14
GOAL_LINE = LENGTH / 2 - 4.0  # IIHF: goal line 4 m from the end boards
CIRCLE_R = 4.5


def arc(cx, cy, r, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def boards(length=LENGTH, width=WIDTH, corner=CORNER):
    hx, hy = length / 2, width / 2
    parts = [
        np.array([[-hx + corner, hy], [hx - corner, hy]]),
        arc(hx - corner, hy - corner, corner, np.pi / 2, 0),
        np.array([[hx, hy - corner], [hx, -hy + corner]]),
        arc(hx - corner, -hy + corner, corner, 0, -np.pi / 2),
        np.array([[hx - corner, -hy], [-hx + corner, -hy]]),
        arc(-hx + corner, -hy + corner, corner, -np.pi / 2, -np.pi),
        np.array([[-hx, -hy + corner], [-hx, hy - corner]]),
        arc(-hx + corner, hy - corner, corner, np.pi, np.pi / 2),
    ]
    return np.concatenate(parts)


def half_width(x, length=LENGTH, width=WIDTH, corner=CORNER):
    hx, hy = length / 2, width / 2
    c = hx - corner
    a = np.abs(x)
    return np.where(a <= c, hy, hy - corner + np.sqrt(np.maximum(corner**2 - (a - c) ** 2, 0)))


def vline(x, n=40):
    h = float(half_width(x))
    return np.stack([np.full(n, x), np.linspace(-h, h, n)], 1)


def circle(cx, cy, r=CIRCLE_R, n=72):
    return arc(cx, cy, r, 0, 2 * np.pi, n)


def lines(width=WIDTH):
    """name -> world polyline"""
    out = {
        "boards": boards(width=width),
        "centre": vline(0),
        "blue_l": vline(-BLUE_LINE),
        "blue_r": vline(BLUE_LINE),
        "goal_l": vline(-GOAL_LINE),
        "goal_r": vline(GOAL_LINE),
        "centre_circle": circle(0, 0),
    }
    for sx in (-20, 20):
        for sy in (-7, 7):
            out[f"fo_{sx}_{sy}"] = circle(sx, sy)
    return out


NEUTRAL_DOTS = [(sx * 5.64, sy * 7.0) for sx in (-1, 1) for sy in (-1, 1)]


def goal_frame(sign, depth=1.12, half=0.915, height=1.22):
    """3D wireframe polylines of a goal at x = sign*GOAL_LINE (mouth facing the centre)."""
    x0, x1 = sign * GOAL_LINE, sign * (GOAL_LINE + depth)
    pts = lambda *p: np.array(p, float)
    return [
        pts((x0, -half, 0), (x0, -half, height), (x0, half, height), (x0, half, 0)),
        pts((x1, -half, 0), (x1, -half, height), (x1, half, height), (x1, half, 0)),
        pts((x0, -half, height), (x1, -half, height)),
        pts((x0, half, height), (x1, half, height)),
        pts((x0, -half, 0), (x1, -half, 0)),
        pts((x0, half, 0), (x1, half, 0)),
    ]


def crease(sign, r=1.8):
    return (
        arc(sign * GOAL_LINE, 0, r, np.pi / 2, 3 * np.pi / 2 if sign > 0 else -np.pi / 2, 24)
        if sign > 0
        else arc(sign * GOAL_LINE, 0, r, -np.pi / 2, np.pi / 2, 24)
    )
