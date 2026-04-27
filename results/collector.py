import json
import os
import datetime
import config


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _timestamp():
    return datetime.datetime.utcnow().isoformat() + "Z"


def save_result(algorithm, backend_name, counts, validation, metadata=None):
    """
    Saves a single algorithm run result to results/<algorithm>/<timestamp>.json.
    """
    _ensure_dir(os.path.join(config.RESULTS_DIR, algorithm))

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{algorithm}_{backend_name}_{ts}.json"
    filepath = os.path.join(config.RESULTS_DIR, algorithm, filename)

    record = {
        "algorithm": algorithm,
        "backend": backend_name,
        "timestamp": _timestamp(),
        "shots": config.SHOTS,
        "counts": counts,
        "validation": validation,
        "metadata": metadata or {},
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    return filepath


def save_vqe_result(backend_name, vqe_results, validation, metadata=None):
    """Saves VQE multi-iteration results."""
    _ensure_dir(os.path.join(config.RESULTS_DIR, "vqe"))

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"vqe_{backend_name}_{ts}.json"
    filepath = os.path.join(config.RESULTS_DIR, "vqe", filename)

    record = {
        "algorithm": "vqe",
        "backend": backend_name,
        "timestamp": _timestamp(),
        "shots": config.SHOTS,
        "iterations": vqe_results,
        "validation": validation,
        "metadata": metadata or {},
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    return filepath


def load_latest_result(algorithm, backend_name):
    """Loads the most recent result file for a given algorithm and backend."""
    result_dir = os.path.join(config.RESULTS_DIR, algorithm)
    if not os.path.exists(result_dir):
        return None

    files = [
        f for f in os.listdir(result_dir)
        if f.startswith(f"{algorithm}_{backend_name}") and f.endswith(".json")
    ]
    if not files:
        return None

    files.sort(reverse=True)
    filepath = os.path.join(result_dir, files[0])
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_results(algorithm):
    """Loads all saved results for a given algorithm across all backends."""
    result_dir = os.path.join(config.RESULTS_DIR, algorithm)
    if not os.path.exists(result_dir):
        return []

    records = []
    for filename in sorted(os.listdir(result_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(result_dir, filename), "r") as f:
                records.append(json.load(f))
    return records
