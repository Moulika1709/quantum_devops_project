import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def build_ansatz(n_qubits=2, layers=2, params=None):
    """
    Hardware-efficient ansatz: RY rotations + CX entanglement layers.
    params: list of float angles, length = n_qubits * layers.
            If None, random parameters are used.
    """
    n_params = n_qubits * layers
    if params is None:
        params = np.random.uniform(0, 2 * np.pi, n_params)

    circuit = QuantumCircuit(n_qubits)
    param_idx = 0
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.ry(params[param_idx], q)
            param_idx += 1
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)

    return circuit, params


def get_h2_hamiltonian():
    """
    Simplified H2 molecule Hamiltonian in Pauli representation (2 qubits).
    Coefficients approximate STO-3G basis at equilibrium bond length.
    """
    hamiltonian = SparsePauliOp.from_list([
        ("II", -1.0523732),
        ("IZ",  0.3979374),
        ("ZI", -0.3979374),
        ("ZZ", -0.0112801),
        ("XX",  0.1809312),
    ])
    return hamiltonian


def compute_expectation_value(counts, hamiltonian, n_qubits, shots):
    """
    Compute expectation value <psi|H|psi> from measurement counts.
    Approximates using Z-basis measurements only (simplified for testing).
    """
    energy = 0.0
    total = sum(counts.values())

    for bitstring, count in counts.items():
        prob = count / total
        # map bitstring to eigenvalue of ZZ...Z operator
        z_vals = [1 - 2 * int(b) for b in bitstring]
        eigenvalue = 1.0
        for z in z_vals:
            eigenvalue *= z
        energy += prob * eigenvalue

    return energy


def build_vqe_measurement_circuit(n_qubits=2, layers=2, params=None):
    """
    Builds ansatz + measurement circuit for VQE energy estimation.
    Returns circuit and parameters used.
    """
    circuit, used_params = build_ansatz(n_qubits=n_qubits, layers=layers, params=params)
    circuit.measure_all()
    return circuit, used_params


def run_vqe_iterations(run_fn, n_qubits=2, layers=2, n_iterations=5, shots=1024):
    """
    Runs n_iterations of VQE with random parameter sets.
    run_fn: callable that takes a circuit and returns counts dict.
    Returns list of (params, energy) tuples.
    """
    results = []
    for i in range(n_iterations):
        circuit, params = build_vqe_measurement_circuit(n_qubits=n_qubits, layers=layers)
        counts = run_fn(circuit, shots=shots)
        hamiltonian = get_h2_hamiltonian()
        energy = compute_expectation_value(counts, hamiltonian, n_qubits, shots)
        results.append({
            "iteration": i + 1,
            "params": params.tolist(),
            "energy": energy,
            "counts": counts,
        })
    return results
