import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import config  # noqa: E402
from algorithms.grover import build_grover_circuit  # noqa: E402
from algorithms.qft import build_qft_circuit  # noqa: E402
from validation.statistical import compare_simulator_vs_hardware  # noqa: E402
import backend as bk  # noqa: E402


SHOTS = config.SHOTS

# these tests only run when explicitly targeting hardware or braket
# on local-only runs they compare clean sim vs noisy sim as a proxy
SKIP_REAL_HW = config.BACKEND not in ("braket", "qpu")


def _get_sim_counts(circuit):
    return bk.run_circuit_local(circuit, shots=SHOTS, noise=False)


def _get_noisy_counts(circuit):
    """Noisy local sim acts as hardware proxy when real QPU is not available."""
    return bk.run_circuit_local(circuit, shots=SHOTS, noise=True)


def _get_hardware_counts(circuit):
    if SKIP_REAL_HW:
        return _get_noisy_counts(circuit)
    return bk.run_circuit_braket(circuit, shots=SHOTS)


class TestGroverSimVsHardware:
    def test_grover_fidelity_sim_vs_hardware(self):
        circuit = build_grover_circuit(n_qubits=3, target_state=5)
        sim_counts = _get_sim_counts(circuit)
        hw_counts = _get_hardware_counts(circuit)
        result = compare_simulator_vs_hardware(sim_counts, hw_counts, SHOTS)

        print(f"\nGrover Sim vs Hardware Fidelity: {result['fidelity']}")
        print(f"Mean Absolute Error: {result['mean_absolute_error']}")

        if SKIP_REAL_HW:
            # noisy sim should still achieve reasonable fidelity
            assert result["fidelity"] > 0.70, (
                f"Sim vs noisy-sim fidelity too low: {result['fidelity']}"
            )
        else:
            assert result["fidelity"] >= config.FIDELITY_THRESHOLD, (
                f"Grover fidelity below threshold: {result['fidelity']}"
            )

    def test_grover_comparison_structure(self):
        circuit = build_grover_circuit(n_qubits=3, target_state=5)
        sim_counts = _get_sim_counts(circuit)
        hw_counts = _get_hardware_counts(circuit)
        result = compare_simulator_vs_hardware(sim_counts, hw_counts, SHOTS)

        assert "fidelity" in result
        assert "per_state_absolute_difference" in result
        assert "mean_absolute_error" in result
        assert 0.0 <= result["fidelity"] <= 1.0


class TestQFTSimVsHardware:
    def test_qft_fidelity_sim_vs_hardware(self):
        circuit = build_qft_circuit(n_qubits=4)
        sim_counts = _get_sim_counts(circuit)
        hw_counts = _get_hardware_counts(circuit)
        result = compare_simulator_vs_hardware(sim_counts, hw_counts, SHOTS)

        print(f"\nQFT Sim vs Hardware Fidelity: {result['fidelity']}")

        if SKIP_REAL_HW:
            assert result["fidelity"] > 0.65, (
                f"QFT sim vs noisy-sim fidelity too low: {result['fidelity']}"
            )
        else:
            assert result["fidelity"] >= config.FIDELITY_THRESHOLD, (
                f"QFT fidelity below threshold: {result['fidelity']}"
            )
