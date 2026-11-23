"""Preflight and retrieve only the pinned, approved TP-6 binary assets."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.executer import GIB, RAM_MIN, disk_usage, memory, windows_api

RELEASE = "7dbe640e127602bf506815e822c09758de73c482"
MODEL = f"https://huggingface.co/PaddlePaddle/SLANet_plus_onnx/resolve/{RELEASE}"
MODEL_HASHES = {
    "inference.onnx": "7790c0c13ce064782c9d22ebeb16b4da8216f83d3ba576da962c106ef58386da",
    "inference.yml": "8a6372d3269a6f112fe13a2da7952a84da6e112c10a3146cbb43de5bd01d19fa",
}
WHEELS = {
    "flatbuffers": ("25.12.19", "flatbuffers-25.12.19-py2.py3-none-any.whl"),
    "numpy": ("2.4.2", "numpy-2.4.2-cp313-cp313-win_amd64.whl"),
    "onnxruntime": ("1.32.0", "onnxruntime-1.32.0-cp313-cp313-win_amd64.whl"),
    "opencv-python-headless": (
        "4.13.0.92",
        "opencv_python_headless-4.13.0.92-cp37-abi3-win_amd64.whl",
    ),
    "packaging": ("26.3", "packaging-26.3-py3-none-any.whl"),
    "protobuf": ("7.36.2", "protobuf-7.36.2-cp310-abi3-win_amd64.whl"),
    "pyyaml": ("6.0.3", "pyyaml-6.0.3-cp313-cp313-win_amd64.whl"),
}
HUB = {
    "huggingface.co",
    "us.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co",
    "cas-bridge.xethub.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
}
PYPI = {"files.pythonhosted.org"}
TOTAL_BYTES = 77_630_593
CORPUS = {
    "smoke": (3, "3a268d55a847b7f4c539ee9609fe9aa32dfaf5a0198ebedb04f38ad0125d0f50"),
    "validation": (
        48,
        "2f22782d48006af060c85993eb7f93679fbf784fb13fd79488f776717f40d584",
    ),
}


def write_report(name, data):
    directory = ROOT / "rapports"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def catalogue(root):
    manifest = json.loads(
        (root / "provenance-telechargements.json").read_text(encoding="utf-8")
    )
    rows = manifest["files"]
    models = set()
    wheels = {}
    if manifest["model_revision"] != RELEASE or len(rows) != 9:
        raise ValueError("Révision ou nombre de fichiers inattendu")
    for row in rows:
        file = row["file"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", file) or not re.fullmatch(
            r"[0-9a-f]{64}", row["sha256"]
        ):
            raise ValueError("Nom ou empreinte invalide")
        if (
            type(row["size_bytes"]) is not int
            or not 0 < row["size_bytes"] <= TOTAL_BYTES
        ):
            raise ValueError("Taille d’artefact invalide")
        if row["type"] == "modele":
            if (
                file in models
                or MODEL_HASHES.get(file) != row["sha256"]
                or row["url"] != f"{MODEL}/{file}"
            ):
                raise ValueError("Poids ou configuration non autorisés")
            models.add(file)
        elif row["type"] == "wheels":
            name = row["package"].lower()
            if name in wheels or WHEELS.get(name) != (row["version"], file):
                raise ValueError("Roue non autorisée")
            u = urlsplit(row["url"])
            if (
                u.netloc not in PYPI
                or u.scheme != "https"
                or u.query
                or u.fragment
                or u.path.split("/")[-1] != file
            ):
                raise ValueError("Source PyPI non autorisée")
            wheels[name] = (row["version"], row["sha256"])
        else:
            raise ValueError("Type non autorisé")
    if models != MODEL_HASHES.keys() or wheels.keys() != WHEELS.keys():
        raise ValueError("Liste incomplète")
    if (
        sum(row["size_bytes"] for row in rows) != TOTAL_BYTES
        or manifest["total_bytes"] != TOTAL_BYTES
    ):
        raise ValueError("Budget de téléchargement non conforme")
    lock = {}
    for line in (
        (root / "requirements-cpu.lock").read_text(encoding="utf-8").splitlines()
    ):
        matched = re.fullmatch(
            r"([a-zA-Z0-9_-]+)==([0-9.]+) --hash=sha256:([0-9a-f]{64})", line
        )
        if not matched or matched[1].lower() in lock:
            raise ValueError("Verrou pip invalide")
        lock[matched[1].lower()] = (matched[2], matched[3])
    if lock != wheels:
        raise ValueError("Les empreintes pip et le manifeste divergent")
    return rows


def verify_file(path, row):
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != row["size_bytes"]
        or digest(path) != row["sha256"]
    ):
        raise ValueError(f"SHA-256 ou taille invalide : {row['file']}")


class ApprovedRedirects(HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts = hosts

    def check(self, url):
        u = urlsplit(url)
        if u.scheme != "https" or u.netloc not in self.hosts:
            raise ValueError("Hôte HTTPS non autorisé")

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_one(root, row, expires):
    directory = root / row["type"]
    directory.mkdir(exist_ok=True)
    path = directory / row["file"]
    if path.exists():
        verify_file(path, row)
        return
    allowed = ApprovedRedirects(HUB if row["type"] == "modele" else PYPI)
    allowed.check(row["url"])
    opener = build_opener(allowed)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with (
            opener.open(
                Request(
                    row["url"],
                    headers={
                        "Accept-Encoding": "identity",
                        "User-Agent": "DocFlow-TP6/1.0",
                    },
                ),
                timeout=25,
            ) as response,
            temporary.open("xb") as output,
        ):
            allowed.check(response.geturl())
            if response.status != 200:
                raise ValueError("Réponse HTTP inattendue")
            size = 0
            while block := response.read(128 * 1024):
                size += len(block)
                if size > row["size_bytes"] or time.monotonic() > expires:
                    raise ValueError(
                        "Taille ou durée maximale de téléchargement dépassée"
                    )
                output.write(block)
        verify_file(temporary, row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_corpus(root):
    summary = {}
    for split, (size, expected_hash) in CORPUS.items():
        directory = root / "corpus" / split
        references = directory / "references.json"
        if references.is_symlink() or digest(references) != expected_hash:
            raise ValueError(f"Annotations modifiées : {split}")
        cases = json.loads(references.read_text(encoding="utf-8"))
        files = set()
        for case in cases:
            name = case["file"]
            if not re.fullmatch(r"[a-z0-9_-]+\.png", name) or name in files:
                raise ValueError("Entrée de corpus invalide")
            files.add(name)
            path = directory / name
            if path.is_symlink() or digest(path) != case["sha256"]:
                raise ValueError(f"Corpus modifié : {split}/{name}")
        if len(files) != size or len({case["id"] for case in cases}) != size:
            raise ValueError("Effectif du corpus incorrect")
        summary[split] = {"count": size, "references_sha256": expected_hash}
    return summary


def preflight(report):
    if (
        os.name != "nt"
        or platform.machine().upper() != "AMD64"
        or sys.version_info[:2] != (3, 13)
    ):
        raise SystemExit("Windows et CPython 3.13 x64 requis.")
    kernel, _ = windows_api()
    ram = memory(kernel)
    free_disk = shutil.disk_usage(ROOT).free
    result = {
        "os": platform.platform(),
        "windows_image": os.environ.get("ImageVersion"),
        "logical_cpu": os.cpu_count(),
        "python": platform.python_version(),
        "ram_total_bytes": ram.total,
        "ram_available_bytes": ram.available,
        "ram_minimum_bytes": RAM_MIN,
        "disk_free_bytes": free_disk,
        "disk_pack_bytes": disk_usage(),
        "corpus": verify_corpus(ROOT),
    }
    result["accepted"] = bool(
        os.cpu_count()
        and os.cpu_count() >= 2
        and ram.available >= RAM_MIN
        and free_disk >= GIB
        and result["disk_pack_bytes"] <= GIB
    )
    write_report(report, result)
    print(f"RAM disponible : {ram.available / GIB:.2f} Gio. Seuil : 2 Gio.", flush=True)
    if not result["accepted"]:
        raise SystemExit(f"Précontrôle refusé : voir {report}. Aucun benchmark lancé.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "fetch", "verify", "installed"])
    parser.add_argument("--avant-installation", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    reports = {
        "preflight": "avant-installation.json"
        if args.avant_installation
        else "precontrole.json",
        "fetch": "avant-telechargement.json",
        "installed": "avant-benchmark.json",
    }
    if mode in reports:
        preflight(reports[mode])
    rows = catalogue(ROOT)
    if mode == "preflight":
        return
    status = []
    if mode == "verify":
        verify_corpus(ROOT)
    if mode == "installed":
        from importlib.metadata import version

        if (
            sys.prefix == sys.base_prefix
            or Path(sys.prefix).resolve() != (ROOT / ".venv-tp6").resolve()
        ):
            raise SystemExit("Le venv isolé TP‑6 est requis.")
        if any(version(name) != spec[0] for name, spec in WHEELS.items()):
            raise SystemExit("Dépendances installées non conformes.")
    expires = time.monotonic() + 180
    for row in rows:
        if mode == "fetch":
            fetch_one(ROOT, row, expires)
        else:
            verify_file(ROOT / row["type"] / row["file"], row)
        status.append({key: row[key] for key in ("file", "sha256", "size_bytes")})
    if mode == "fetch":
        uncompressed = 0
        for entry in (r for r in rows if r["type"] == "wheels"):
            with ZipFile(ROOT / entry["type"] / entry["file"]) as wheel:
                uncompressed += sum(item.file_size for item in wheel.infolist())
        estimate = disk_usage() + 2 * uncompressed + 80 * (2**20)
        write_report(
            "telechargements.json",
            {
                "revision": RELEASE,
                "actual_bytes": TOTAL_BYTES,
                "projected_disk_bytes": estimate,
                "verified": status,
            },
        )
        if estimate > GIB or estimate - disk_usage() > shutil.disk_usage(ROOT).free:
            raise SystemExit("Budget disque insuffisant pour l’installation.")
    print(f"{len(status)} fichiers vérifiés, {TOTAL_BYTES} octets.")


if __name__ == "__main__":
    main()
