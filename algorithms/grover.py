from qiskit import QuantumCircuit
import math


def build_oracle(n_qubits, target_state):
    """
    Phase oracle that marks the target_state with a phase flip.
    target_state: integer index of the target basis state.
    """
    oracle = QuantumCircuit(n_qubits)
    target_bits = format(target_state, f"0{n_qubits}b")

    # flip qubits where target bit is 0 so all are |1> at target
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            oracle.x(i)

    # multi-controlled Z via H + multi-controlled X + H
    oracle.h(n_qubits - 1)
    oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)

    # undo bit flips
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            oracle.x(i)

    return oracle


def build_diffuser(n_qubits):
    """Grover diffusion operator (inversion about average)."""
    diffuser = QuantumCircuit(n_qubits)
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def build_grover_circuit(n_qubits=3, target_state=5):
    """
    Full Grover circuit for n_qubits searching for target_state.
    Optimal number of iterations: floor(pi/4 * sqrt(2^n)).
    """
    n_states = 2 ** n_qubits
    n_iterations = max(1, math.floor(math.pi / 4 * math.sqrt(n_states)))

    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target_state)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True)

    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def expected_top_state(n_qubits, target_state):
    """Returns the bitstring expected to have highest probability."""
    return format(target_state, f"0{n_qubits}b")
