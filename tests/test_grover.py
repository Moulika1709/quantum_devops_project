import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import pytest  # noqa: E402
import config  # noqa: E402
from algorithms.grover import build_grover_circuit, expected_top_state  # noqa: E402
from validation.statistical import validate_grover  # noqa: E402
from results.collector import save_result  # noqa: E402
import backend as bk  # noqa: E402


N_QUBITS = 3
TARGET_STATE = 5
SHOTS = config.SHOTS


@pytest.fixture(scope="module")
def grover_counts():
    circuit = build_grover_circuit(n_qubits=N_QUBITS, target_state=TARGET_STATE)
    counts = bk.run_circuit(circuit, shots=SHOTS)
    return counts


def test_grover_circuit_builds():
    circuit = build_grover_circuit(n_qubits=N_QUBITS, target_state=TARGET_STATE)
    assert circuit is not None
    assert circuit.num_qubits == N_QUBITS


def test_grover_has_measurements():
    circuit = build_grover_circuit(n_qubits=N_QUBITS, target_state=TARGET_STATE)
    assert circuit.num_clbits == N_QUBITS


def test_grover_counts_not_empty(grover_counts):
    assert len(grover_counts) > 0
    assert sum(grover_counts.values()) == SHOTS


def test_grover_top_state_matches_target(grover_counts):
    top_state = max(grover_counts, key=grover_counts.get)
    expected = expected_top_state(N_QUBITS, TARGET_STATE)
    assert top_state == expected, (
        f"Expected top state {expected}, got {top_state}. "
        f"Counts: {grover_counts}"
    )


def test_grover_target_probability_above_threshold(grover_counts):
    total = sum(grover_counts.values())
    expected = expected_top_state(N_QUBITS, TARGET_STATE)
    target_prob = grover_counts.get(expected, 0) / total
    # Grover should amplify target well above uniform (1/8 = 0.125 for 3 qubits)
    assert target_prob > 0.5, (
        f"Target state probability {target_prob:.4f} too low. "
        f"Expected > 0.50 for {N_QUBITS}-qubit Grover."
    )


def test_grover_chi_squared(grover_counts):
    validation = validate_grover(grover_counts, TARGET_STATE, N_QUBITS, SHOTS)
    chi = validation["chi_squared"]
    if chi["passed"] is None:
        pytest.skip("Insufficient counts for chi-squared test")
    assert chi["passed"], (
        f"Chi-squared test failed: stat={chi['statistic']}, p={chi['p_value']}"
    )


def test_grover_validation_and_save(grover_counts):
    validation = validate_grover(grover_counts, TARGET_STATE, N_QUBITS, SHOTS)
    filepath = save_result(
        algorithm="grover",
        backend_name=config.BACKEND,
        counts=grover_counts,
        validation=validation,
        metadata={"n_qubits": N_QUBITS, "target_state": TARGET_STATE},
    )
    assert os.path.exists(filepath)


def test_grover_noisy_vs_noiseless():
    """Compares noiseless vs noisy local simulator - relevant for H3 (hardware noise)."""
    circuit = build_grover_circuit(n_qubits=N_QUBITS, target_state=TARGET_STATE)
    counts_clean = bk.run_circuit_local(circuit, shots=SHOTS, noise=False)
    counts_noisy = bk.run_circuit_local(circuit, shots=SHOTS, noise=True)

    expected = expected_top_state(N_QUBITS, TARGET_STATE)
    total_clean = sum(counts_clean.values())
    total_noisy = sum(counts_noisy.values())

    prob_clean = counts_clean.get(expected, 0) / total_clean
    prob_noisy = counts_noisy.get(expected, 0) / total_noisy

    # noisy simulator should have lower target probability than clean
    assert prob_clean >= prob_noisy, (
        f"Noise model did not reduce target probability: "
        f"clean={prob_clean:.4f}, noisy={prob_noisy:.4f}"
    )
