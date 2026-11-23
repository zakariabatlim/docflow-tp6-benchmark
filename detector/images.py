from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np


from .grid import allowed_shape

MAX_SIDE = 1800
MIN_TABLE_AREA = 0.07


@dataclass
class GridLayout:
    ink: np.ndarray
    xs: list[int]
    ys: list[int]
    gray: np.ndarray | None = None
    to_source: np.ndarray | None = None
    warning: str = ""

    @property
    def width(self):
        return len(self.xs) - 1

    @property
    def height(self):
        return len(self.ys) - 1


def _threshold(gray):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )


def _rectified_candidates(gray, ink):
    """Propose des contours fermés et leur redressement, par surface décroissante."""
    height, width = gray.shape
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(contour) < width * height * MIN_TABLE_AREA:
            break
        polygon = cv2.approxPolyDP(contour, 0.022 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        boundary = cv2.boundingRect(polygon)
        left, right = np.split(polygon.reshape(4, 2)[np.argsort(polygon[:, 0, 0])], 2)
        tl, bl = left[np.argsort(left[:, 1])]
        tr, br = right[np.argsort(right[:, 1])]
        target_width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        target_height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if min(target_width, target_height) < 40:
            continue
        margin = 10
        target = np.float32(
            [
                [margin, margin],
                [margin + target_width, margin],
                [margin + target_width, margin + target_height],
                [margin, margin + target_height],
            ]
        )
        transform = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), target)
        image = cv2.warpPerspective(
            gray,
            transform,
            (target_width + 2 * margin, target_height + 2 * margin),
            borderValue=255,
        )
        yield image, boundary, np.linalg.inv(transform)


def _is_fragment(bounds, enclosing):
    x, y, w, h = bounds
    left, top, width, height = enclosing
    return (
        w * h < 0.35 * width * height
        and left - 3 <= x
        and top - 3 <= y
        and x + w <= left + width + 3
        and y + h <= top + height + 3
    )


def _rectified_photo(gray, ink):
    candidate = next(_rectified_candidates(gray, ink), None)
    return candidate[0] if candidate is not None else None


def _line_positions(mask, axis):
    coverage = np.count_nonzero(mask, axis=axis)
    stride = mask.shape[axis]
    indices = np.flatnonzero(coverage >= max(stride * 0.16, coverage.max() * 0.44))
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
    return [int(np.average(group, weights=coverage[group])) for group in groups]


def _axis_lines(ink, length, horizontal):
    gap = max(3, length // 120)
    span = max(13, length // 18)
    close_shape = (1, gap) if horizontal else (gap, 1)
    open_shape = (1, span) if horizontal else (span, 1)
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones(close_shape, np.uint8))
    return cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones(open_shape, np.uint8))


def _has_weak_segment(line_mask, positions, divisions, radius):
    # Une interruption locale brève est tolérée. Un bord de cellule absent ne l'est pas.
    for coordinate in positions[1:-1]:
        band = line_mask[max(0, coordinate - radius) : coordinate + radius + 1]
        stroke = band.any(axis=0)
        for left, right in pairwise(divisions):
            margin = max(1, (right - left) // 10)
            if stroke[left + margin : right - margin].mean() < 0.48:
                return True
    return False


def _layout(ink, *, tolerate_drift=False, gray=None, to_source=None):
    height, width = ink.shape
    horizontal = _axis_lines(ink, width, horizontal=True)
    vertical = _axis_lines(ink, height, horizontal=False)
    if tolerate_drift:
        # Après redressement, le tracé papier peut encore s'écarter de quelques pixels.
        xs = _line_positions(cv2.dilate(vertical, np.ones((1, 9), np.uint8)), axis=0)
        ys = _line_positions(cv2.dilate(horizontal, np.ones((9, 1), np.uint8)), axis=1)
    else:
        xs = _line_positions(vertical, axis=0)
        ys = _line_positions(horizontal, axis=1)
    if len(xs) < 2 or len(ys) < 2:
        return None

    # Les axes d'une grille rectangulaire se croisent, contrairement aux traits isolés.
    joint_size = 17 if tolerate_drift else 7
    joint_kernel = np.ones((joint_size, joint_size), np.uint8)
    joint = cv2.bitwise_and(
        cv2.dilate(horizontal, joint_kernel),
        cv2.dilate(vertical, joint_kernel),
    )
    intersections = joint[np.ix_(ys, xs)] != 0
    if tolerate_drift:
        # Un trait dans une cellule peut apparaître dans la projection mais ne
        # rencontre presque jamais les séparateurs de l'autre direction.
        rows = intersections.mean(axis=1) >= 0.33
        columns = intersections.mean(axis=0) >= 0.33
        ys = [position for position, keep in zip(ys, rows, strict=True) if keep]
        xs = [position for position, keep in zip(xs, columns, strict=True) if keep]
        intersections = intersections[np.ix_(rows, columns)]
    if not allowed_shape(len(ys) - 1, len(xs) - 1):
        return None
    if min(np.diff(xs)) < 8 or min(np.diff(ys)) < 8:
        return None
    if (
        not intersections[[0, -1], :][:, [0, -1]].all()
        or (intersections.mean(axis=0) < 0.75).any()
        or (intersections.mean(axis=1) < 0.75).any()
    ):
        return None
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    text = cv2.bitwise_and(ink, cv2.bitwise_not(lines))
    radius = 10 if tolerate_drift else 5
    warning = ""
    if _has_weak_segment(horizontal, ys, xs, radius) or _has_weak_segment(
        vertical.T, xs, ys, radius
    ):
        warning = (
            "Certains séparateurs sont trop incomplets. Vérifiez les dimensions et "
            "recopiez les valeurs, notamment si la page contient des cellules fusionnées."
        )
    return GridLayout(text, xs, ys, gray=gray, to_source=to_source, warning=warning)


def _overlap(first, second):
    x, y, w, h = first
    a, b, width, height = second
    return x < a + width and a < x + w and y < b + height and b < y + h


def detect_grid(gray):
    """Détecte les séparateurs d'un tableau unique, avec un essai après redressement."""
    if min(gray.shape) < 32:
        return None
    ink = _threshold(gray)
    candidates = []
    direct = _layout(ink, gray=gray)
    if direct is not None:
        area = (direct.xs[-1] - direct.xs[0]) * (direct.ys[-1] - direct.ys[0])
        if area >= gray.size * MIN_TABLE_AREA:
            candidates.append(direct)

    independent, enclosures = [], []
    for rectified, boundary, to_source in _rectified_candidates(gray, ink):
        rectified_ink = _threshold(rectified)
        options = {"gray": rectified, "to_source": to_source}
        layout = _layout(rectified_ink, **options) or _layout(
            rectified_ink, tolerate_drift=True, **options
        )
        size = layout.height * layout.width if layout is not None else 0
        nested = any(
            _is_fragment(boundary, previous) and (size <= 1 or previous_size > 1)
            for previous, previous_size in enclosures
        )
        if nested:
            continue
        enclosures.append((boundary, size))
        if layout is None:
            continue
        candidates.append(layout)
        if size >= 2 and not any(
            _overlap(boundary, previous) for previous in independent
        ):
            independent.append(boundary)

    best = max(
        candidates,
        key=lambda layout: (layout.width * layout.height, not layout.warning),
        default=None,
    )
    if best is not None and len(independent) > 1:
        best.warning = (
            "Plusieurs tableaux possibles sont présents sur la page. "
            "Recadrez un seul tableau ou corrigez la grille manuellement."
        )
    return best
