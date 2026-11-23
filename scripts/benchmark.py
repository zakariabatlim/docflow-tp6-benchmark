"""Compare two fixed pipelines over the frozen TP-6 corpus."""

import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAMP = "TP6-SUPERVISED-WORKER"
if (
    __name__ == "__main__"
    and "--self-test" not in sys.argv
    and sys.stdin.readline().strip() != STAMP
):
    raise SystemExit("Utiliser scripts/executer.py pour appliquer les limites Windows.")

import cv2
import numpy as np
import yaml
from detector.errors import ValidationError
from detector.geometry import Geometry
from detector.images import MAX_SIDE, detect_grid
from scripts.slanet import MODEL, SLANet, decode, labels, layout_cells

cv2.setNumThreads(1)
THRESHOLD = 0.80


def project(points, transform):
    return cv2.perspectiveTransform(np.float32([points]), transform)[0].tolist()


def origin_quad(geom, shape):
    return (np.float32(geom.corners) * [shape[1] - 1, shape[0] - 1]).astype(np.float32)


def opencv_cells(geom, shape):
    points = origin_quad(geom, shape)
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), points
    )
    cells = []
    for r, (top, bottom) in enumerate(zip(geom.y, geom.y[1:])):
        for c, (left, right) in enumerate(zip(geom.x, geom.x[1:])):
            quad = project(
                [[left, top], [right, top], [right, bottom], [left, bottom]], transform
            )
            cells.append({"row": r, "col": c, "rowspan": 1, "colspan": 1, "quad": quad})
    return {"dims": [len(geom.y) - 1, len(geom.x) - 1], "cells": cells}


def proposal(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    layout = detect_grid(gray)
    if layout is None:
        return None, "grille absente"
    try:
        geom = Geometry.from_layout(layout, gray.shape)
        geom.validate_image(gray.shape)
    except (ValidationError, cv2.error):
        return None, "géométrie invalide"
    return geom, layout.warning


def rectified(bgr, quad):
    src = np.float32(quad)
    width = (
        round(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))) + 1
    )
    height = (
        round(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))) + 1
    )
    if min(width, height) < 8 or max(width, height) > MAX_SIDE * 2:
        raise ValueError("Zone hors limites")
    dest = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    transform = cv2.getPerspectiveTransform(src, dest)
    crop = cv2.warpPerspective(
        bgr, transform, (width, height), borderValue=(255, 255, 255)
    )
    return crop, np.linalg.inv(transform)


def detect(bgr, mode, engine, reference_zone):
    source = bgr
    to_source = None
    if reference_zone is not None:
        source, to_source = rectified(bgr, reference_zone)
    # Crop arm receives the same supplied region. Automatic arm independently
    # proposes a region with the current detector, then falls back to the page.
    if mode == "opencv" or reference_zone is None:
        geom, warning = proposal(source)
    else:
        geom, warning = None, ""
    if mode == "opencv":
        if geom is None or warning:
            return None, warning or "grille absente"
        result = opencv_cells(geom, source.shape)
    else:
        if warning.startswith("Plusieurs tableaux"):
            return None, warning
        if geom is not None:
            source, to_source = rectified(source, origin_quad(geom, source.shape))
        try:
            result = engine(source)
        except ValueError as exc:
            return None, str(exc)
        if not any(c["quad"] for c in result["cells"]):
            return None, "aucune cellule localisée"
    if to_source is not None:
        for cell in result["cells"]:
            if cell["quad"]:
                cell["quad"] = project(cell["quad"], to_source)
    return result, ""


def key(cell):
    return tuple(cell[name] for name in ("row", "col", "rowspan", "colspan"))


def iou(one, two):
    if one is None or two is None:
        return 0.0
    a, b = np.float32(one), np.float32(two)
    area_one, area_two = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    total = area_one + area_two - intersection
    return float(intersection / total) if total > 0 else 0.0


def evaluate(ref, pred):
    reference = {key(cell): cell for cell in ref.get("cells", [])}
    detected = {key(cell): cell for cell in pred["cells"]} if pred else {}
    overlaps = [
        iou(cell["quad"], detected[k]["quad"]) if k in detected else 0.0
        for k, cell in reference.items()
    ]
    matches = [
        k
        for k, overlap in zip(reference, overlaps, strict=True)
        if overlap >= THRESHOLD
    ]
    fused = lambda keys: sum(k[2] > 1 or k[3] > 1 for k in keys)
    return {
        "accepted": bool(pred),
        "correct_reject": not pred and ref["expected"] == "reject",
        "exact_dims": bool(pred and pred["dims"] == ref.get("dims")),
        "cells_reference": len(reference),
        "cells_predicted": len(detected),
        "cells_iou_ge_080": len(matches),
        "iou_sum_reference": sum(overlaps),
        "fusions_reference": fused(reference),
        "fusions_predicted": fused(detected),
        "fusions_correct": fused(matches),
        "structure_exact": bool(
            pred and reference and detected.keys() == reference.keys()
        ),
    }


def ocr_cells(bgr, pred, exe):
    crops = []
    keys = []
    for cell in (pred or {}).get("cells", []):
        if not cell["quad"]:
            continue
        # Order convex polygon points consistently before rectification.
        points = np.float32(cell["quad"])
        middle = points.mean(axis=0)
        order = np.argsort(
            np.arctan2(points[:, 1] - middle[1], points[:, 0] - middle[0])
        )
        points = points[order]
        points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
        try:
            cut, _ = rectified(bgr, points)
        except ValueError:
            continue
        dx = max(2, min(7, round(cut.shape[1] * 0.04)))
        dy = max(2, min(7, round(cut.shape[0] * 0.07)))
        if min(cut.shape[1] - 2 * dx, cut.shape[0] - 2 * dy) < 5:
            continue
        patch = cv2.cvtColor(cut[dy:-dy, dx:-dx], cv2.COLOR_BGR2GRAY)
        crops.append(
            cv2.copyMakeBorder(patch, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
        )
        keys.append(key(cell))
    if not crops:
        return {}
    ok, data = cv2.imencodemulti(".tiff", crops)
    if not ok:
        raise RuntimeError("Encodage TIFF OCR indisponible")
    run = subprocess.run(
        [str(exe), "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"],
        input=data.tobytes(),
        capture_output=True,
        check=True,
        timeout=12,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    output = {}
    for row in csv.DictReader(
        io.StringIO(run.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    ):
        if row["level"] == "5" and row["text"].strip():
            index = int(row["page_num"]) - 1
            if 0 <= index < len(keys):
                output.setdefault(keys[index], []).append(row["text"].strip())
    return {k: " ".join(words) for k, words in output.items()}


def evaluate_ocr(ref, ocr):
    totals = {
        "text_exact": 0,
        "text_reference": 0,
        "number_exact": 0,
        "number_reference": 0,
        "blank_false": 0,
        "blank_reference": 0,
    }
    for cell in ref.get("cells", []):
        wanted = cell["text"]
        seen = ocr.get(key(cell), "")
        if not wanted:
            totals["blank_reference"] += 1
            totals["blank_false"] += bool(seen)
            continue
        totals["text_reference"] += 1
        totals["text_exact"] += wanted == seen
        if re.fullmatch(r"\d+(?:,\d+)?", wanted):
            totals["number_reference"] += 1
            totals["number_exact"] += wanted == seen.replace(".", ",")
    return totals


def timings(values):
    times = sorted(values)
    return (
        {
            "docs": len(times),
            "median_s": statistics.median(times),
            "p95_s": times[math.ceil(0.95 * len(times)) - 1],
        }
        if times
        else None
    )


def metrics(rows):
    positive = [row for row in rows if row["expected"] == "table"]
    negatives = [row for row in rows if row["expected"] == "reject"]
    totals = {
        field: sum(row[field] for row in rows)
        for field in (
            "cells_iou_ge_080",
            "cells_reference",
            "cells_predicted",
            "fusions_correct",
            "fusions_reference",
            "fusions_predicted",
        )
    }
    row = {
        "tables_reference": len(positive),
        "dims_exact": sum(r["exact_dims"] for r in positive),
        "topology_exact": sum(r["structure_exact"] for r in positive),
        "non_tables_reference": len(negatives),
        "correct_rejects": sum(r["correct_reject"] for r in negatives),
        **totals,
        "precision_cells": totals["cells_iou_ge_080"]
        / max(1, totals["cells_predicted"]),
        "recall_cells": totals["cells_iou_ge_080"] / max(1, totals["cells_reference"]),
        "mean_iou_reference": sum(r["iou_sum_reference"] for r in positive)
        / max(1, totals["cells_reference"]),
        "processing": timings([r["total_s"] for r in rows]),
        "structure": timings([r["structure_s"] for r in rows]),
    }
    successful_ocr = [r["ocr"] for r in positive if r["ocr"] is not None]
    row["ocr"] = (
        {field: sum(r[field] for r in successful_ocr) for field in successful_ocr[0]}
        if successful_ocr
        else None
    )
    row["ocr_docs"] = len(successful_ocr)
    row["ocr_errors"] = sum(r["ocr_error"] is not None for r in rows)
    return row


def run(args):
    split = ROOT / "corpus" / "validation"
    records = json.loads((split / "references.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    engine = SLANet() if args.mode == "slanet" else None
    startup = time.perf_counter() - started
    rows = []
    for ref in records:
        if args.arm == "zone" and ref["expected"] == "reject":
            continue
        started = time.perf_counter()
        bgr = cv2.imdecode(
            np.frombuffer((split / ref["file"]).read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if bgr is None:
            raise ValueError(f"Image invalide : {ref['id']}")
        h, w = bgr.shape[:2]
        new_factor = min(1, MAX_SIDE / max(w, h))
        if new_factor < 1:
            bgr = cv2.resize(
                bgr,
                (round(w * new_factor), round(h * new_factor)),
                interpolation=cv2.INTER_AREA,
            )
        scale = np.float32([w / bgr.shape[1], h / bgr.shape[0]])
        supplied = (
            (np.float32(ref["zone"]) / scale).tolist() if args.arm == "zone" else None
        )
        prediction, reject = detect(bgr, args.mode, engine, supplied)
        structure_time = time.perf_counter() - started
        ocr, ocr_error = None, None
        if args.tesseract:
            try:
                ocr = evaluate_ocr(ref, ocr_cells(bgr, prediction, args.tesseract))
            except (subprocess.SubprocessError, RuntimeError, cv2.error) as exc:
                ocr_error = type(exc).__name__
        total = time.perf_counter() - started
        if prediction:
            for cell in prediction["cells"]:
                if cell["quad"]:
                    cell["quad"] = (np.array(cell["quad"]) * scale).tolist()
        rows.append(
            {
                "id": ref["id"],
                "kind": ref["kind"],
                "expected": ref["expected"],
                "expected_dims": ref.get("dims"),
                "prediction": prediction,
                "rejection": reject or None,
                "structure_s": structure_time,
                "total_s": total,
                "ocr": ocr,
                "ocr_error": ocr_error,
                **evaluate(ref, prediction),
            }
        )
    report = {
        "mode": args.mode,
        "arm": args.arm,
        "startup_s": startup,
        "first_document_s": rows[0]["total_s"],
        "ocr_engine": args.ocr_version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "onnxruntime": version("onnxruntime") if engine else None,
        "documents": rows,
        "metrics": metrics(rows),
        "by_category": {
            category: metrics([r for r in rows if r["kind"] == category])
            for category in sorted({r["kind"] for r in rows})
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def self_test():
    dims, cells = layout_cells(
        [
            "<thead>",
            "<tr>",
            "<td",
            ' colspan="2"',
            ">",
            "</td>",
            "</tr>",
            "</thead>",
            "<tbody>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "</tbody>",
        ]
    )
    assert dims == [2, 2] and [key(c) for c in cells] == [
        (0, 0, 1, 2),
        (1, 0, 1, 1),
        (1, 1, 1, 1),
    ]
    try:
        layout_cells(["<tr>", "<td", ' rowspan="2"', ">", "</td>", "</tr>"])
    except ValueError:
        pass
    else:
        raise AssertionError("Fusion hors grille acceptée")
    vocabulary = labels(
        yaml.safe_load((MODEL / "inference.yml").read_text(encoding="utf-8"))
    )
    assert len(vocabulary) == 50
    tokens = ("sos", "<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>", "eos")
    scores = np.zeros((1, len(tokens), 50), np.float32)
    boxes = np.zeros((1, len(tokens), 8), np.float32)
    for index, token in enumerate(tokens):
        scores[0, index, vocabulary.index(token)] = 1
    boxes[0, 3] = [0.1, 0.1, 0.9, 0.1, 0.9, 0.8, 0.1, 0.8]
    decoded = decode(boxes, scores, vocabulary, 100, (90, 100))
    assert decoded["dims"] == [1, 1] and decoded["cells"][0]["quad"][0] == [10, 10]
    assert timings([1, 2, 100])["p95_s"] == 100
    assert (
        abs(
            iou(
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 0], [10, 0], [10, 10], [0, 10]],
            )
            - 1
        )
        < 1e-6
    )
    smoke = ROOT / "corpus" / "smoke"
    records = json.loads((smoke / "references.json").read_text(encoding="utf-8"))
    ref = records[0]
    image = cv2.imdecode(
        np.frombuffer((smoke / ref["file"]).read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    result, why = detect(image, "opencv", None, None)
    assert evaluate(ref, result)["exact_dims"], why
    assert cv2.imencodemulti(
        ".tiff", [np.zeros((16, 16), np.uint8), np.ones((20, 20), np.uint8)]
    )[0]
    print("Contrôles locaux réussis, sans chargement du modèle ONNX.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=["opencv", "slanet"])
    parser.add_argument("--arm", choices=["photo", "zone"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument("--ocr-version")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.mode and args.arm and args.output:
        run(args)
    else:
        parser.error("Mode, bras et sortie requis.")
