import math
from qiskit import QuantumCircuit


def build_qft_circuit(n_qubits=4, inverse=False):
    """
    Quantum Fourier Transform on n_qubits.
    If inverse=True, builds QFT-dagger (used in phase estimation etc.).
    Input state: |0...0> by default; prepend state prep as needed.
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)

    # prepare a non-trivial input: alternating X gates so state is not all-zero
    for i in range(0, n_qubits, 2):
        circuit.x(i)

    circuit.barrier()

    def qft_core(qc, qubits):
        n = len(qubits)
        for i in range(n):
            qc.h(qubits[i])
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                qc.cp(angle, qubits[j], qubits[i])
        # bit-reversal permutation
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - i - 1])

    qubits = list(range(n_qubits))

    if inverse:
        # build forward then invert
        qft_sub = QuantumCircuit(n_qubits)
        qft_core(qft_sub, qubits)
        inverse_qft = qft_sub.inverse()
        circuit.compose(inverse_qft, inplace=True)
    else:
        qft_core(circuit, qubits)

    circuit.barrier()
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def qft_expected_distribution(n_qubits):
    """
    For a uniform superposition input through QFT, output should be
    concentrated. Returns None since exact expected state depends on input.
    Validation uses distribution uniformity checks instead.
    """
    return None
