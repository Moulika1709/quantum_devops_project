import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import config  # noqa: E402
import backend as bk  # noqa: E402
from algorithms.grover import build_grover_circuit  # noqa: E402
from algorithms.qft import build_qft_circuit  # noqa: E402
from algorithms.vqe import run_vqe_iterations  # noqa: E402
from validation.statistical import (  # noqa: E402
    validate_grover,
    validate_qft,
    validate_vqe,
    compare_simulator_vs_hardware,
)
from results.collector import save_result, save_vqe_result  # noqa: E402
from reports.generator import generate_json_report, generate_html_report  # noqa: E402


GROVER_QUBITS = 3
GROVER_TARGET = 5
QFT_QUBITS = 4
VQE_QUBITS = 2
VQE_LAYERS = 2
VQE_ITERATIONS = 5


def run_grover():
    print(f"[Grover] Running on backend: {config.BACKEND}")
    circuit = build_grover_circuit(n_qubits=GROVER_QUBITS, target_state=GROVER_TARGET)
    counts = bk.run_circuit(circuit, shots=config.SHOTS)
    validation = validate_grover(counts, GROVER_TARGET, GROVER_QUBITS, config.SHOTS)
    path = save_result(
        algorithm="grover",
        backend_name=config.BACKEND,
        counts=counts,
        validation=validation,
        metadata={"n_qubits": GROVER_QUBITS, "target_state": GROVER_TARGET},
    )
    print(f"[Grover] Saved: {path}")
    print(f"[Grover] Top state match: {validation['top_state_accuracy']['match']}")
    print(f"[Grover] Target prob: {validation['observed_target_probability']}")
    return counts, validation


def run_qft():
    print(f"[QFT] Running on backend: {config.BACKEND}")
    circuit = build_qft_circuit(n_qubits=QFT_QUBITS)
    counts = bk.run_circuit(circuit, shots=config.SHOTS)
    validation = validate_qft(counts, QFT_QUBITS, config.SHOTS)
    path = save_result(
        algorithm="qft",
        backend_name=config.BACKEND,
        counts=counts,
        validation=validation,
        metadata={"n_qubits": QFT_QUBITS},
    )
    print(f"[QFT] Saved: {path}")
    print(f"[QFT] Non-trivial output: {validation['non_trivial_output']}")
    print(f"[QFT] Entropy ratio: {validation['entropy_ratio']}")
    return counts, validation


def run_vqe():
    print(f"[VQE] Running on backend: {config.BACKEND}")

    def _run_fn(circuit, shots=None):
        return bk.run_circuit(circuit, shots=shots or config.SHOTS)

    results = run_vqe_iterations(
        run_fn=_run_fn,
        n_qubits=VQE_QUBITS,
        layers=VQE_LAYERS,
        n_iterations=VQE_ITERATIONS,
        shots=config.SHOTS,
    )
    validation = validate_vqe(results)
    path = save_vqe_result(
        backend_name=config.BACKEND,
        vqe_results=results,
        validation=validation,
        metadata={"n_qubits": VQE_QUBITS, "layers": VQE_LAYERS},
    )
    print(f"[VQE] Saved: {path}")
    print(f"[VQE] Min energy: {validation['min_energy']}, Mean: {validation['mean_energy']}")
    return results, validation


def run_comparison():
    """Runs simulator and noisy simulator, compares distributions."""
    print("[Comparison] Running simulator vs noisy simulator")
    comparison = {}

    grover_circuit = build_grover_circuit(n_qubits=GROVER_QUBITS, target_state=GROVER_TARGET)
    g_sim = bk.run_circuit_local(grover_circuit, shots=config.SHOTS, noise=False)
    g_noisy = bk.run_circuit_local(grover_circuit, shots=config.SHOTS, noise=True)
    comparison["grover"] = compare_simulator_vs_hardware(g_sim, g_noisy, config.SHOTS)
    print(f"[Comparison] Grover fidelity: {comparison['grover']['fidelity']}")

    qft_circuit = build_qft_circuit(n_qubits=QFT_QUBITS)
    q_sim = bk.run_circuit_local(qft_circuit, shots=config.SHOTS, noise=False)
    q_noisy = bk.run_circuit_local(qft_circuit, shots=config.SHOTS, noise=True)
    comparison["qft"] = compare_simulator_vs_hardware(q_sim, q_noisy, config.SHOTS)
    print(f"[Comparison] QFT fidelity: {comparison['qft']['fidelity']}")

    return comparison


def main():
    print("=" * 60)
    print("DevOps Quantum Testing Framework - Full Run")
    print(f"Backend: {config.BACKEND} | Shots: {config.SHOTS}")
    print("=" * 60)

    grover_counts, grover_val = run_grover()
    qft_counts, qft_val = run_qft()
    vqe_results, vqe_val = run_vqe()
    comparison = run_comparison()

    all_results = {
        "grover": [{
            "backend": config.BACKEND,
            "shots": config.SHOTS,
            "timestamp": "",
            "counts": grover_counts,
            "validation": grover_val,
        }],
        "qft": [{
            "backend": config.BACKEND,
            "shots": config.SHOTS,
            "timestamp": "",
            "counts": qft_counts,
            "validation": qft_val,
        }],
        "vqe": [{
            "backend": config.BACKEND,
            "shots": config.SHOTS,
            "timestamp": "",
            "iterations": vqe_results,
            "validation": vqe_val,
        }],
    }

    json_path = generate_json_report(all_results, comparison=comparison)
    html_path = generate_html_report(all_results, comparison=comparison)

    print("\n" + "=" * 60)
    print("Run Complete")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
