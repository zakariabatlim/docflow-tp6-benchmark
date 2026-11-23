"""Adaptateur CPU pour le modèle SLANet_plus ONNX épinglé."""

import re
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import yaml
from detector.grid import MAX_COLUMNS, MAX_ROWS, allowed_shape

MODEL = Path(__file__).resolve().parents[1] / "modele"
REQUIRED_OUTPUTS = ("fetch_name_0", "fetch_name_1")


def labels(config):
    post = config["PostProcess"]
    if post["name"] != "TableLabelDecode" or not post["merge_no_span_structure"]:
        raise ValueError("Décodeur inattendu")
    chars = [token for token in post["character_dict"] if token != "<td>"]
    if "<td></td>" not in chars:
        chars.append("<td></td>")
    return ["sos", *chars, "eos"]


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current = None
        self.open_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            if self.current is not None or len(self.rows) >= MAX_ROWS:
                raise ValueError("Ligne imbriquée ou limite de lignes dépassée")
            self.current = []
        elif tag == "td":
            if (
                self.current is None
                or self.open_cell
                or len(self.current) >= MAX_COLUMNS
            ):
                raise ValueError("Cellule hors ligne ou mal formée")
            attrs_dict = dict(attrs)
            if len(attrs_dict) != len(attrs) or set(attrs_dict) - {
                "rowspan",
                "colspan",
            }:
                raise ValueError("Attribut inattendu")
            spans = [attrs_dict.get(key, "1") for key in ("rowspan", "colspan")]
            if not all(re.fullmatch(r"[1-9][0-9]?", value or "") for value in spans):
                raise ValueError("Fusion non valide")
            self.current.append(tuple(map(int, spans)))
            self.open_cell = True
        elif tag not in {"thead", "tbody"}:
            raise ValueError("Balise inattendue")

    def handle_endtag(self, tag):
        if tag == "td" and self.open_cell:
            self.open_cell = False
        elif tag == "tr" and self.current is not None and not self.open_cell:
            self.rows.append(self.current)
            self.current = None
        elif tag not in {"thead", "tbody"}:
            raise ValueError("Fermeture de balise inattendue")


def layout_cells(tokens):
    parser = TableParser()
    parser.feed("".join(tokens))
    parser.close()
    if parser.current is not None or parser.open_cell or not parser.rows:
        raise ValueError("Table incomplète")
    occupied = set()
    cells = []
    for row, entries in enumerate(parser.rows):
        col = 0
        for rowspan, colspan in entries:
            while (row, col) in occupied:
                col += 1
            if row + rowspan > len(parser.rows) or col + colspan > MAX_COLUMNS:
                raise ValueError("Fusion hors grille")
            covered = {
                (r, c)
                for r in range(row, row + rowspan)
                for c in range(col, col + colspan)
            }
            if covered & occupied:
                raise ValueError("Occupation conflictuelle")
            occupied.update(covered)
            cells.append(
                {"row": row, "col": col, "rowspan": rowspan, "colspan": colspan}
            )
            col += colspan
    width = max((c + 1 for _, c in occupied), default=0)
    if (
        not allowed_shape(len(parser.rows), width)
        or len(occupied) != len(parser.rows) * width
    ):
        raise ValueError("Grille trouée ou hors limites")
    return [len(parser.rows), width], cells


def decode(boxes, probs, vocabulary, scale, shape):
    if (
        boxes.ndim != 3
        or probs.ndim != 3
        or boxes.shape[:2] != probs.shape[:2]
        or boxes.shape[0] != 1
        or boxes.shape[-1] != 8
        or probs.shape[-1] != len(vocabulary)
    ):
        raise RuntimeError("Tenseurs SLANet_plus inattendus")
    if not np.isfinite(boxes).all() or not np.isfinite(probs).all():
        raise ValueError("Tenseurs non finis")
    tokens = []
    quads = []
    for index, (number, coordinates) in enumerate(
        zip(probs[0].argmax(axis=1), boxes[0], strict=True)
    ):
        label = vocabulary[int(number)]
        if label == "eos":
            break
        if label == "sos":
            if index != 0:
                raise ValueError("Nouvelle séquence imprévue")
            continue
        tokens.append(label)
        if label in {"<td", "<td></td>"}:
            points = coordinates.reshape(4, 2) * scale
            points = np.clip(points, 0, [shape[1] - 1, shape[0] - 1]).astype(np.float32)
            if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 8:
                # The model can provide a structural token (notably a blank cell) without geometry.
                quads.append(None)
            else:
                quads.append(points.tolist())
    dims, cells = layout_cells(tokens)
    if len(quads) != len(cells):
        raise ValueError("Correspondance structure et polygones incohérente")
    for cell, quad in zip(cells, quads, strict=True):
        cell["quad"] = quad
    return {"dims": dims, "cells": cells}


class SLANet:
    def __init__(self):
        import onnxruntime as ort

        config = yaml.safe_load((MODEL / "inference.yml").read_text(encoding="utf-8"))
        if config["Global"]["model_name"] != "SLANet_plus":
            raise ValueError("Configuration de modèle inattendue")
        ops = {
            key: value
            for op in config["PreProcess"]["transform_ops"]
            for key, value in op.items()
        }
        self.edge = ops["ResizeTableImage"]["max_len"]
        self.mean = np.float32(ops["NormalizeImage"]["mean"])
        self.std = np.float32(ops["NormalizeImage"]["std"])
        if (
            self.edge != 488
            or ops["PaddingTableImage"]["size"] != [488, 488]
            or ops["DecodeImage"]["img_mode"] != "BGR"
            or ops["NormalizeImage"]["scale"] != "1./255."
        ):
            raise ValueError("Prétraitement inattendu")
        self.vocabulary = labels(config)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(MODEL / "inference.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if [entry.name for entry in self.session.get_inputs()] != ["x"] or tuple(
            entry.name for entry in self.session.get_outputs()
        ) != REQUIRED_OUTPUTS:
            raise RuntimeError("Signature ONNX inattendue")
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Fournisseur d’inférence inattendu")

    def __call__(self, bgr):
        height, width = bgr.shape[:2]
        ratio = self.edge / max(width, height)
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        data = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        data = (data / np.float32(255) - self.mean) / self.std
        padded = np.zeros((self.edge, self.edge, 3), dtype=np.float32)
        padded[: size[1], : size[0]] = data
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])
        boxes, probs = self.session.run(list(REQUIRED_OUTPUTS), {"x": tensor})
        return decode(
            boxes, probs, self.vocabulary, max(width, height), (height, width)
        )
