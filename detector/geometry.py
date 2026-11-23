"""Rectangular table topology in a projective plane on the oriented source photo."""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

import cv2
import numpy as np
from .errors import ValidationError

from .grid import MAX_COLUMNS, MAX_ROWS, allowed_shape
from .images import MAX_SIDE, GridLayout, _axis_lines, _threshold

MIN_INTERVAL = 0.001
MIN_CELL_PIXELS = 8
QUAD_ERROR = "Placez les quatre coins dans l’ordre haut gauche, haut droite, bas droite, bas gauche."


def _coordinate(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not isfinite(value):
        raise ValidationError("Les coordonnées doivent être comprises entre 0 et 1.")
    return round(value, 7)


def _axes(values, maximum):
    if not isinstance(values, list) or not 2 <= len(values) <= maximum + 1:
        raise ValidationError("Nombre de séparateurs hors limites.")
    axis = tuple(_coordinate(value) for value in values)
    if (
        axis[0] != 0
        or axis[-1] != 1
        or any(right - left < MIN_INTERVAL for left, right in pairwise(axis))
    ):
        raise ValidationError(
            "Les séparateurs doivent rester distincts et dans l’ordre."
        )
    return axis


@dataclass(frozen=True)
class Geometry:
    corners: tuple
    x: tuple
    y: tuple

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {"corners", "x", "y"}:
            raise ValidationError("La zone du tableau est incomplète.")
        points = raw["corners"]
        if (
            not isinstance(points, list)
            or len(points) != 4
            or any(not isinstance(point, list) or len(point) != 2 for point in points)
        ):
            raise ValidationError(QUAD_ERROR)
        corners = tuple(
            tuple(_coordinate(component) for component in point) for point in points
        )
        for i in range(4):
            a, b, c = (corners[(i + offset) % 4] for offset in range(3))
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross < 0.0001:
                raise ValidationError(QUAD_ERROR)
        xs, ys = _axes(raw["x"], MAX_COLUMNS), _axes(raw["y"], MAX_ROWS)
        if not allowed_shape(len(ys) - 1, len(xs) - 1):
            raise ValidationError("La grille contient trop de cellules.")
        return cls(corners, xs, ys)

    @classmethod
    def from_layout(cls, layout, source_shape):
        xs, ys = layout.xs, layout.ys
        left, right, top, bottom = xs[0], xs[-1], ys[0], ys[-1]
        points = np.float32(
            [[[left, top], [right, top], [right, bottom], [left, bottom]]]
        )
        if layout.to_source is not None:
            points = cv2.perspectiveTransform(points, layout.to_source)
        height, width = source_shape
        return cls.parse(
            {
                "corners": np.clip(points[0] / [width - 1, height - 1], 0, 1).tolist(),
                "x": [(value - left) / (right - left) for value in xs],
                "y": [(value - top) / (bottom - top) for value in ys],
            }
        )

    def as_dict(self):
        return {
            "corners": [list(point) for point in self.corners],
            "x": list(self.x),
            "y": list(self.y),
        }

    def blank_cells(self):
        return [[""] * (len(self.x) - 1) for _ in range(len(self.y) - 1)]

    def _plane(self, source_shape):
        height, width = source_shape
        source = np.float32(self.corners) * np.float32([width - 1, height - 1])
        tl, tr, br, bl = source
        target_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        target_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        scale = min(1.0, (MAX_SIDE - 1) / max(target_width, target_height, 1))
        w, h = round(target_width * scale), round(target_height * scale)
        # A wide outer bound may hide a nearly collapsed short side.
        if (
            min(np.linalg.norm(b - a) for a, b in pairwise([tl, tr, br, bl, tl]))
            < MIN_CELL_PIXELS
        ):
            raise ValidationError(
                "La zone sélectionnée est trop petite pour découper les cellules."
            )
        xs = [round(value * w) for value in self.x]
        ys = [round(value * h) for value in self.y]
        if min(np.diff(xs)) < MIN_CELL_PIXELS or min(np.diff(ys)) < MIN_CELL_PIXELS:
            raise ValidationError("Écartez les séparateurs trop proches dans la photo.")
        to_source = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), source
        )
        axes = np.stack(np.meshgrid(self.x, self.y), axis=-1).astype(np.float32)
        mesh = cv2.perspectiveTransform(axes.reshape(1, -1, 2), to_source).reshape(
            axes.shape
        )
        horizontal = np.linalg.norm(np.diff(mesh, axis=1), axis=2)
        vertical = np.linalg.norm(np.diff(mesh, axis=0), axis=2)
        if min(horizontal.min(), vertical.min()) < MIN_CELL_PIXELS:
            raise ValidationError(
                "Certaines cellules sont trop étroites sur la photo d’origine."
            )
        return source, xs, ys

    def validate_image(self, source_shape):
        self._plane(source_shape)

    def cell_layout(self, gray):
        source, xs, ys = self._plane(gray.shape)
        width, height = xs[-1], ys[-1]
        transform = cv2.getPerspectiveTransform(
            source, np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        )
        aligned = cv2.warpPerspective(
            gray, transform, (width + 1, height + 1), borderValue=255
        )
        ink = _threshold(aligned)
        # Supprimer les traits compatibles avec les séparateurs retenus.
        # Le pixel original dans la zone du texte reste disponible à l'OCR.
        for positions, axis_ink, target in (
            (ys, _axis_lines(ink, width + 1, True), ink),
            (xs, _axis_lines(ink, height + 1, False).T, ink.T),
        ):
            for position in positions:
                band = slice(max(0, position - 4), position + 5)
                target[band] &= cv2.bitwise_not(axis_ink[band])
        return GridLayout(ink, xs, ys, gray=aligned, to_source=np.linalg.inv(transform))


def detect_geometry(gray, detector):
    """Turn the detector's plane into coordinates of the original image."""
    try:
        layout = detector(gray)
        if layout is not None:
            geometry = Geometry.from_layout(layout, gray.shape)
            geometry.validate_image(gray.shape)
            return geometry, layout.warning
    except (ValidationError, cv2.error):
        pass
    return (
        None,
        "Zone de tableau à délimiter. Sélectionnez un tableau sur la photo ou saisissez les cellules.",
    )
