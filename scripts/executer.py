"""Windows-only memory and CPU supervisor for the isolated benchmark."""

import argparse
import ctypes as c
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from ctypes import wintypes as w
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
DISK_LIMIT = GIB
RAM_MIN = 2 * GIB
WORKER_RAM_MAX = GIB
C = c.c_size_t
D = w.DWORD
Q = c.c_int64


class Memory(c.Structure):
    _fields_ = [("length", D), ("load", D)] + [
        (name, c.c_uint64)
        for name in (
            "total",
            "available",
            "page",
            "pavailable",
            "virtual",
            "vavailable",
            "extended",
        )
    ]


class ProcessMemory(c.Structure):
    _fields_ = [("cb", D), ("faults", D)] + [
        (name, C)
        for name in (
            "peak_working",
            "working",
            "peak_paged",
            "paged",
            "peak_nonpaged",
            "nonpaged",
            "page",
            "peak_page",
            "private",
        )
    ]


class BasicLimits(c.Structure):
    _fields_ = [
        ("process_time", Q),
        ("job_time", Q),
        ("flags", D),
        ("minimum", C),
        ("maximum", C),
        ("active", D),
        ("affinity", C),
        ("priority", D),
        ("scheduling", D),
    ]


class ExtendedLimits(c.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", c.c_uint64 * 6),
        ("process_limit", C),
        ("job_limit", C),
        ("process_peak", C),
        ("job_peak", C),
    ]


class Accounting(c.Structure):
    _fields_ = [
        ("user", Q),
        ("kernel", Q),
        ("period_user", Q),
        ("period_kernel", Q),
        ("faults", D),
        ("total", D),
        ("active", D),
        ("terminated", D),
    ]


class ProcessList(c.Structure):
    _fields_ = [("assigned", D), ("included", D), ("ids", C * 32)]


def windows_api():
    kernel = c.WinDLL("kernel32", use_last_error=True)
    psapi = c.WinDLL("psapi", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, D]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
    kernel.QueryInformationJobObject.argtypes = [
        w.HANDLE,
        c.c_int,
        c.c_void_p,
        D,
        c.c_void_p,
    ]
    kernel.OpenProcess.argtypes = [D, w.BOOL, D]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.c_void_p, D]
    return kernel, psapi


def require(ok):
    if not ok:
        raise c.WinError(c.get_last_error())


def memory(kernel):
    result = Memory()
    result.length = c.sizeof(result)
    require(kernel.GlobalMemoryStatusEx(c.byref(result)))
    return result


def disk_usage():
    return sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())


def information(kernel, job, kind, obj):
    require(
        kernel.QueryInformationJobObject(job, kind, c.byref(obj), c.sizeof(obj), None)
    )
    return obj


def working_sets(kernel, psapi, job):
    members = information(kernel, job, 3, ProcessList())
    current = peak_for_one = 0
    for pid in members.ids[: members.included]:
        handle = kernel.OpenProcess(0x0410, False, pid)
        if not handle:
            continue  # A subprocess may exit between the snapshot and this call.
        try:
            details = ProcessMemory()
            details.cb = c.sizeof(details)
            if psapi.GetProcessMemoryInfo(handle, c.byref(details), details.cb):
                current += details.working
                peak_for_one = max(peak_for_one, details.peak_working)
        finally:
            kernel.CloseHandle(handle)
    return current, peak_for_one


def watch(kernel, psapi, job, worker, deadline=None):
    peak = 0
    minimum_free = memory(kernel).available
    stop = None
    started = time.monotonic()
    while True:
        current, peak_one = working_sets(kernel, psapi, job)
        peak = max(peak, current, peak_one)
        remaining = memory(kernel).available
        minimum_free = min(minimum_free, remaining)
        if peak > WORKER_RAM_MAX:
            stop = "plafond de RAM résidente dépassé"
        elif remaining < GIB // 2:
            stop = "pression mémoire Windows, moins de 0,5 Gio disponible"
        elif deadline is not None and time.monotonic() >= deadline:
            stop = "budget de temps global dépassé"
        elif time.monotonic() - started > 480:
            stop = "délai de 480 secondes dépassé"
        if stop:
            require(kernel.TerminateJobObject(job, 1))
            worker.wait(timeout=10)
            break
        if worker.poll() is not None:
            break
        time.sleep(0.05)
    return peak, minimum_free, stop


def run_one(kernel, psapi, mode, arm, directory, ocr, ocr_version, deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError("Budget de temps épuisé avant la passe suivante")
    state = memory(kernel)
    if state.available < RAM_MIN or disk_usage() > DISK_LIMIT:
        raise RuntimeError("Précondition de RAM ou de disque devenue insuffisante")
    report = directory / f"{mode}-{arm}.json"
    args = [
        sys.executable,
        "-I",
        "-B",
        str(ROOT / "scripts" / "benchmark.py"),
        "--mode",
        mode,
        "--arm",
        arm,
        "--output",
        str(report),
    ]
    if ocr:
        args += ["--tesseract", str(ocr), "--ocr-version", ocr_version]
    job = kernel.CreateJobObjectW(None, None)
    require(job)
    limits = ExtendedLimits()
    # Hard ceiling on private committed memory for this worker and its children.
    limits.basic.flags = 0x200 | 0x2000
    limits.job_limit = WORKER_RAM_MAX
    worker = None
    begin = time.perf_counter()
    try:
        require(
            kernel.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits))
        )
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "2",
            "OPENCV_FOR_THREADS_NUM": "1",
        }
        log = directory / f"{mode}-{arm}.log"
        with log.open("wb") as stream:
            worker = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            require(kernel.AssignProcessToJobObject(job, int(worker._handle)))
            worker.stdin.write(b"TP6-SUPERVISED-WORKER\n")
            worker.stdin.close()
            peak, minimum_free, stop = watch(kernel, psapi, job, worker, deadline)
        accounting = information(kernel, job, 1, Accounting())
        limits = information(kernel, job, 9, ExtendedLimits())
        telemetry = {
            "mode": mode,
            "arm": arm,
            "exit_code": worker.returncode,
            "stop_reason": stop,
            "elapsed_s": time.perf_counter() - begin,
            "peak_sampled_working_set_bytes": peak,
            "peak_job_commit_bytes": limits.job_peak,
            "minimum_available_system_bytes": minimum_free,
            "cpu_user_s": accounting.user / 10**7,
            "cpu_kernel_s": accounting.kernel / 10**7,
            "disk_pack_bytes_after": disk_usage(),
            "measurement_ms": 50,
        }
        (directory / f"{mode}-{arm}-ressources.json").write_text(
            json.dumps(telemetry, indent=2) + "\n", encoding="utf-8"
        )
        if (
            stop
            or worker.returncode
            or not report.is_file()
            or telemetry["disk_pack_bytes_after"] > DISK_LIMIT
        ):
            raise RuntimeError(
                f"Essai interrompu : {mode}-{arm}. Consulter le journal et les ressources."
            )
        return telemetry
    finally:
        kernel.CloseHandle(job)
        if worker is not None and worker.poll() is None:
            # Only our own waiting worker, e.g. if assignment to the job failed.
            worker.terminate()
            worker.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument("--deadline-seconds", type=int, default=660)
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 660:
        parser.error("Le budget de temps doit être compris entre 1 et 660 secondes.")
    if os.name != "nt" or sys.version_info[:2] != (3, 13) or c.sizeof(c.c_void_p) != 8:
        raise SystemExit("Windows x64 et CPython 3.13 sont requis.")
    kernel, psapi = windows_api()
    status = memory(kernel)
    if status.available < RAM_MIN or disk_usage() > DISK_LIMIT:
        raise SystemExit(
            "Précontrôle bloqué : il faut 2 Gio de RAM disponible et un pack de moins de 1 Gio."
        )
    output = ROOT / "rapports" / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    # Optional OCR is caller-provided and its metadata is collected separately.
    v_ocr = None
    if args.tesseract:
        if not args.tesseract.is_absolute() or not args.tesseract.is_file():
            raise SystemExit("Chemin absolu du Tesseract local requis.")
        # Keep metadata calls short. The budgeted OCR itself runs in the worker job.
        probe = subprocess.run(
            [str(args.tesseract), "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        v_ocr = probe.stdout.decode("utf-8", errors="replace").splitlines()[0]
    machine = {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu": os.cpu_count(),
        "total_RAM_bytes": status.total,
        "available_at_start_bytes": status.available,
        "machine_type": "github-hosted"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "windows",
        "versions": {
            package: version(package)
            for package in (
                "onnxruntime",
                "numpy",
                "opencv-python-headless",
                "PyYAML",
                "flatbuffers",
                "protobuf",
                "packaging",
            )
        },
    }
    (output / "machine.json").write_text(
        json.dumps(machine, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    deadline = time.monotonic() + args.deadline_seconds
    for arm in ("photo", "zone"):
        for mode in ("opencv", "slanet"):
            results.append(
                run_one(
                    kernel, psapi, mode, arm, output, args.tesseract, v_ocr, deadline
                )
            )
            print(f"{mode} / {arm} terminé", flush=True)
    (output / "ressources.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
