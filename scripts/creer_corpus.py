"""Génère deux jeux synthétiques indépendants à partir de primitives OpenCV."""

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "corpus"
TYPES = (
    "bordures",
    "incline",
    "fusion",
    "sans_bordures",
    "fusion_sans_bordures",
    "traits_partiels",
)
NEGATIVES = ("blanc", "texte", "deux_tables", "tronque")
WORDS = (
    "CAHIER",
    "STYLO",
    "CLASSEUR",
    "PAPIER",
    "DOSSIER",
    "CARTE",
    "BOITE",
    "SAC",
    "REGLE",
)


def quad(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def projection(points, transform):
    return np.round(
        cv2.perspectiveTransform(np.float32([points]), transform)[0], 3
    ).tolist()


def table(kind, rng):
    rows, cols = rng.randint(4, 8), rng.randint(3, 6)
    x = np.r_[0, np.cumsum([rng.randint(85, 140) for _ in range(cols)])]
    y = np.r_[0, np.cumsum([rng.randint(34, 59) for _ in range(rows)])]
    width, height = int(x[-1]), int(y[-1])
    local = np.full((height + 1, width + 1, 3), 247, np.uint8)
    merges = {(0, 0): (1, 2), (2, 1): (2, 1)} if "fusion" in kind else {}
    occupied, cells, strokes = set(), [], set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in occupied:
                continue
            dr, dc = merges.get((r, c), (1, 1))
            occupied.update((i, j) for i in range(r, r + dr) for j in range(c, c + dc))
            l, t, p, b = map(int, (x[c], y[r], x[c + dc], y[r + dr]))
            corners = quad(l, t, p, b)
            if "sans_bordures" not in kind:
                points = [tuple(pt) for pt in corners]
                for a, z in zip(points, points[1:] + points[:1], strict=True):
                    strokes.add(tuple(sorted((a, z))))
            if r == 0:
                value = ("PRODUIT", "CODE", "QTE", "PRIX", "MONTANT", "TOTAL")[c]
            elif c == 0:
                value = rng.choice(WORDS)
            elif rng.random() < 0.18:
                value = ""
            elif c < 3:
                value = str(rng.randint(1, 899))
            else:
                value = f"{rng.randint(1, 99)},{rng.randint(0, 99):02d}"
            scale = rng.uniform(0.49, 0.68)
            cv2.putText(
                local,
                value,
                (l + 8, (t + b) // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (31, 29, 28),
                1,
                cv2.LINE_AA,
            )
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "rowspan": dr,
                    "colspan": dc,
                    "quad": corners,
                    "text": value,
                }
            )
    for a, b in sorted(strokes):
        if kind == "traits_partiels" and rng.random() < 0.45:
            left = tuple(round(u + (v - u) * 0.35) for u, v in zip(a, b, strict=True))
            right = tuple(round(u + (v - u) * 0.50) for u, v in zip(a, b, strict=True))
            cv2.line(local, a, left, (65, 65, 65), 1, cv2.LINE_AA)
            cv2.line(local, right, b, (65, 65, 65), 1, cv2.LINE_AA)
        else:
            cv2.line(local, a, b, (49, 49, 49), 2, cv2.LINE_AA)
    margin = rng.randint(27, 43)
    destination = np.float32(quad(margin, margin, width + margin, height + margin))
    if kind == "incline":
        for i in range(4):
            destination[i] += (rng.randint(-15, 15), rng.randint(-17, 17))
    transform = cv2.getPerspectiveTransform(
        np.float32(quad(0, 0, width, height)), destination
    )
    output = cv2.warpPerspective(
        local,
        transform,
        (width + 2 * margin, height + 2 * margin),
        borderValue=(247, 247, 247),
    )
    if kind == "incline":
        output = cv2.GaussianBlur(output, (3, 3), 0.75)
        darken = np.linspace(0.87, 1.0, output.shape[1], dtype=np.float32)
        output = np.round(output * darken[None, :, None]).astype(np.uint8)
    for cell in cells:
        cell["quad"] = projection(cell["quad"], transform)
    outer = projection(quad(-4, -4, width + 4, height + 4), transform)
    return output, {
        "dims": [rows, cols],
        "zone": outer,
        "cells": cells,
        "expected": "table",
    }


def negative(kind, rng):
    if kind == "tronque":
        image, _ = table("bordures", rng)
        return image[:, : image.shape[1] // 2]
    if kind == "deux_tables":
        a, _ = table("bordures", rng)
        b, _ = table("bordures", rng)
        output = np.full(
            (max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1] + 40, 3),
            247,
            np.uint8,
        )
        output[: a.shape[0], : a.shape[1]] = a
        output[: b.shape[0], a.shape[1] + 40 :] = b
        return output
    output = np.full((315, 550, 3), 247, np.uint8)
    if kind == "texte":
        for i in range(7):
            cv2.putText(
                output,
                f"NOTE INTERNE {rng.randint(10, 99)}   CONTROLE",
                (27, 40 + i * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (32, 32, 32),
                1,
                cv2.LINE_AA,
            )
    return output


def generate(split, specifications, seed):
    directory = ROOT / split
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    for kind, count in specifications:
        for i in range(count):
            identifier = f"{kind}-{i + 1:02}"
            path = directory / f"{identifier}.png"
            if kind in NEGATIVES:
                image, ref = negative(kind, rng), {"expected": "reject"}
            else:
                image, ref = table(kind, rng)
            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 7])
            if not ok:
                raise RuntimeError(identifier)
            data = encoded.tobytes()
            path.write_bytes(data)
            records.append(
                {
                    "id": identifier,
                    "kind": kind,
                    "file": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    **ref,
                }
            )
    (directory / "references.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate("smoke", [("bordures", 1), ("fusion", 1), ("texte", 1)], 1061)
    generate(
        "validation",
        [(kind, 6) for kind in TYPES] + [(kind, 3) for kind in NEGATIVES],
        20261122,
    )
