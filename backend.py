import config


def get_local_backend(noise=False):
    from qiskit_aer import AerSimulator
    if noise:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        noise_model = NoiseModel()
        error_1q = depolarizing_error(0.01, 1)
        error_2q = depolarizing_error(0.02, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "h", "x"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        return AerSimulator(noise_model=noise_model)
    return AerSimulator()


def run_circuit_local(circuit, shots=None, noise=False):
    from qiskit import transpile
    shots = shots or config.SHOTS
    backend = get_local_backend(noise=noise)
    transpiled = transpile(circuit, backend)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def run_circuit_braket(circuit, shots=None):
    from qiskit_braket_provider import BraketProvider
    shots = shots or config.SHOTS

    provider = BraketProvider()
    backend = provider.get_backend("SV1")
    from qiskit import transpile
    transpiled = transpile(circuit, backend)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def run_circuit(circuit, shots=None, backend_override=None, noise=False):
    backend = backend_override or config.BACKEND
    if backend == "local":
        return run_circuit_local(circuit, shots=shots, noise=noise)
    elif backend in ("braket", "qpu"):
        return run_circuit_braket(circuit, shots=shots)
    else:
        raise ValueError(f"Unknown backend: {backend}")
