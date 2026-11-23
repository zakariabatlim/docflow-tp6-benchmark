"""Export only measured data and clearly represent partial runs."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "rapports"
PASSES = {(mode, arm) for mode in ("opencv", "slanet") for arm in ("photo", "zone")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_report(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dimensions(shape):
    return "x".join(map(str, shape)) if shape else None


def render(root):
    root.mkdir(exist_ok=True)
    passes, rows = {}, []
    for path in sorted(root.glob("*/*-ressources.json")):
        usage = read(path)
        mode, arm = usage["mode"], usage["arm"]
        identity = mode, arm
        if identity not in PASSES or identity in passes:
            raise ValueError("Passe non prévue ou présente en double")
        path = path.with_name(f"{mode}-{arm}.json")
        doc = (
            read(path)
            if path.is_file() and usage["exit_code"] == 0 and not usage["stop_reason"]
            else None
        )
        if doc and (doc["mode"], doc["arm"]) != identity:
            raise ValueError("Identité de passe incohérente")
        passes[identity] = (doc, usage)
        if not doc:
            continue
        for item in doc["documents"]:
            prediction = item["prediction"]
            rows.append(
                {
                    "moteur": mode,
                    "bras": arm,
                    "document": item["id"],
                    "categorie": item["kind"],
                    "attendu": item["expected"],
                    "dimensions_reference": dimensions(item["expected_dims"]),
                    "dimensions_predites": dimensions(prediction["dims"])
                    if prediction
                    else None,
                    "dimensions_exactes": item["exact_dims"],
                    "topologie_exacte": item["structure_exact"],
                    "cellules_reference": item["cells_reference"],
                    "cellules_predites": item["cells_predicted"],
                    "cellules_iou_ge_080": item["cells_iou_ge_080"],
                    "somme_iou_reference": item["iou_sum_reference"],
                    "fusions_reference": item["fusions_reference"],
                    "fusions_predites": item["fusions_predicted"],
                    "fusions_correctes": item["fusions_correct"],
                    "rejet_correct": item["correct_reject"],
                    "rejet": item["rejection"],
                    "structure_s": item["structure_s"],
                    "total_s": item["total_s"],
                }
            )
    summaries, comparisons = [], {}
    for mode, arm in sorted(PASSES):
        doc, usage = passes.get((mode, arm), (None, {}))
        m = doc["metrics"] if doc else {}
        timers = m.get("processing") or {}
        summaries.append(
            {
                "moteur": mode,
                "bras": arm,
                "etat": "mesuree"
                if doc
                else ("interrompue" if usage else "non_executee"),
                "documents": timers.get("docs"),
                "tables_reference": m.get("tables_reference"),
                "dimensions_exactes": m.get("dims_exact"),
                "topologies_exactes": m.get("topology_exact"),
                "precision_cellules": m.get("precision_cells"),
                "rappel_cellules": m.get("recall_cells"),
                "iou_moyenne": m.get("mean_iou_reference"),
                "fusions_correctes": m.get("fusions_correct"),
                "fusions_reference": m.get("fusions_reference"),
                "rejets_corrects": m.get("correct_rejects"),
                "negatifs_reference": m.get("non_tables_reference"),
                "demarrage_s": doc["startup_s"] if doc else None,
                "mediane_s": timers.get("median_s"),
                "p95_s": timers.get("p95_s"),
                "ram_pic_residente_octets": usage.get("peak_sampled_working_set_bytes"),
                "ram_pic_engagee_octets": usage.get("peak_job_commit_bytes"),
                "ram_systeme_min_octets": usage.get("minimum_available_system_bytes"),
                "cpu_utilisateur_s": usage.get("cpu_user_s"),
                "cpu_noyau_s": usage.get("cpu_kernel_s"),
                "cause_arret": usage.get("stop_reason"),
            }
        )
    for arm, count in (("photo", 48), ("zone", 36)):
        before, _ = passes.get(("opencv", arm), (None, {}))
        after, _ = passes.get(("slanet", arm), (None, {}))
        if not before or not after:
            continue
        reference_fields = (
            "id",
            "kind",
            "expected",
            "expected_dims",
            "cells_reference",
            "fusions_reference",
        )
        left = [tuple(d[k] for k in reference_fields) for d in before["documents"]]
        right = [tuple(d[k] for k in reference_fields) for d in after["documents"]]
        if left != right or len(left) != count or len({r[0] for r in left}) != count:
            raise ValueError("Comparaison impossible : documents différents")
        a, b = before["metrics"], after["metrics"]
        comparisons[arm] = {
            "documents_apparies": count,
            "delta_dimensions_exactes": b["dims_exact"] - a["dims_exact"],
            "delta_precision_cellules": b["precision_cells"] - a["precision_cells"],
            "delta_rappel_cellules": b["recall_cells"] - a["recall_cells"],
            "delta_fusions_correctes": b["fusions_correct"] - a["fusions_correct"],
            "delta_rejets_corrects": b["correct_rejects"] - a["correct_rejects"],
            "delta_mediane_s": b["processing"]["median_s"]
            - a["processing"]["median_s"],
            "delta_p95_s": b["processing"]["p95_s"] - a["processing"]["p95_s"],
        }
    result = {"passes": summaries, "comparaisons": comparisons}
    (root / "synthese.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_report(root / "synthese.csv", summaries)
    if rows:
        csv_report(root / "resultats-par-document.csv", rows)
    print(f"{len(rows)} mesures documentaires exportées.")


if __name__ == "__main__":
    render(ROOT)
