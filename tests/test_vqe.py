import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import config  # noqa: E402
from algorithms.vqe import (  # noqa: E402
    build_vqe_measurement_circuit,
    run_vqe_iterations,
    get_h2_hamiltonian,
)
from validation.statistical import validate_vqe  # noqa: E402
from results.collector import save_vqe_result  # noqa: E402
import backend as bk  # noqa: E402


N_QUBITS = 2
LAYERS = 2
VQE_ITERATIONS = 5
SHOTS = config.SHOTS


def _run_fn(circuit, shots=None):
    return bk.run_circuit(circuit, shots=shots or SHOTS)


@pytest.fixture(scope="module")
def vqe_results():
    return run_vqe_iterations(
        run_fn=_run_fn,
        n_qubits=N_QUBITS,
        layers=LAYERS,
        n_iterations=VQE_ITERATIONS,
        shots=SHOTS,
    )


def test_vqe_circuit_builds():
    circuit, params = build_vqe_measurement_circuit(n_qubits=N_QUBITS, layers=LAYERS)
    assert circuit is not None
    assert circuit.num_qubits == N_QUBITS
    assert len(params) == N_QUBITS * LAYERS


def test_vqe_hamiltonian_is_hermitian():
    h = get_h2_hamiltonian()
    # SparsePauliOp is Hermitian by construction from real coefficients
    matrix = h.to_matrix()
    assert np.allclose(matrix, matrix.conj().T), "Hamiltonian is not Hermitian"


def test_vqe_runs_all_iterations(vqe_results):
    assert len(vqe_results) == VQE_ITERATIONS


def test_vqe_each_iteration_has_counts(vqe_results):
    for result in vqe_results:
        assert "counts" in result
        assert sum(result["counts"].values()) == SHOTS


def test_vqe_energy_is_real_valued(vqe_results):
    for result in vqe_results:
        energy = result["energy"]
        assert isinstance(energy, float), f"Energy is not float: {energy}"
        assert not np.isnan(energy), "Energy is NaN"


def test_vqe_energy_within_physical_bounds(vqe_results):
    """
    For H2 STO-3G, energies should be bounded.
    Minimum eigenvalue is approximately -1.857 Hartree.
    Expectation values from Z-basis measurements stay in [-1, 1] range.
    """
    for result in vqe_results:
        assert -2.0 <= result["energy"] <= 2.0, (
            f"Energy {result['energy']} outside physical bounds [-2, 2]"
        )


def test_vqe_different_params_give_different_energies(vqe_results):
    """Verifies that different parameter sets produce different energy estimates."""
    energies = [r["energy"] for r in vqe_results]
    unique_energies = set(round(e, 4) for e in energies)
    assert len(unique_energies) > 1, (
        "All VQE iterations produced identical energies — parameter variation not working"
    )


def test_vqe_validation_pass(vqe_results):
    validation = validate_vqe(vqe_results)
    assert validation["passed"], (
        f"VQE validation failed. Min energy: {validation['min_energy']}"
    )


def test_vqe_validation_and_save(vqe_results):
    validation = validate_vqe(vqe_results)
    filepath = save_vqe_result(
        backend_name=config.BACKEND,
        vqe_results=vqe_results,
        validation=validation,
        metadata={"n_qubits": N_QUBITS, "layers": LAYERS, "iterations": VQE_ITERATIONS},
    )
    assert os.path.exists(filepath)
