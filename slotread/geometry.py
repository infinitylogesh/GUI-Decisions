"""Boxes and masks: IoU, COCO polygons, and the radial (centre + rays) mask encoding."""

import math

import numpy as np
from PIL import Image, ImageDraw


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def poly_mask(polys, W, H):
    """COCO polygons (flat lists of x, y) -> boolean mask."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i], p[i + 1]) for i in range(0, len(p), 2)], fill=1)
    return np.array(im, dtype=bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union if union else 0.0), int(inter), int(union)


def radial_encode(mask, K):
    """Centre: the point inside the mask farthest from its boundary (a centroid can fall outside a
    non-convex mask). Ray k at angle 2 pi k / K (0 = right, turning clockwise on screen): the distance to
    the farthest mask pixel along it. Returns (cx, cy, d) in pixels."""
    from scipy.ndimage import distance_transform_edt
    H, W = mask.shape
    dt = distance_transform_edt(mask)
    cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
    diag = math.hypot(W, H)
    t = np.arange(0, diag, 0.5)
    ang = 2 * np.pi * np.arange(K) / K
    xs = np.rint(cx + np.outer(np.cos(ang), t)).astype(int)
    ys = np.rint(cy + np.outer(np.sin(ang), t)).astype(int)
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    inside = np.zeros_like(inb)
    inside[inb] = mask[ys[inb], xs[inb]]
    last = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1), 0)
    return float(cx), float(cy), t[last]


def radial_decode(cx, cy, d, W, H):
    """Ray endpoints -> polygon -> boolean mask (all in pixels). Returns (mask, polygon points)."""
    K = len(d)
    ang = 2 * np.pi * np.arange(K) / K
    pts = [(cx + float(r) * math.cos(a), cy + float(r) * math.sin(a)) for r, a in zip(d, ang)]
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.array(im, dtype=bool), pts


def grow(b, g, W, H):
    """A box (x1, y1, x2, y2, pixels) grown by factor g about its centre, clipped to the image."""
    gx, gy = (b[2] - b[0]) * (g - 1) / 2, (b[3] - b[1]) * (g - 1) / 2
    return [max(0, b[0] - gx), max(0, b[1] - gy), min(W - 1, b[2] + gx), min(H - 1, b[3] + gy)]
