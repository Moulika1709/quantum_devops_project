import os

BACKEND = os.environ.get("QUANTUM_BACKEND", "local")

# local   -> qiskit-aer simulator (no AWS required)
# braket  -> AWS Braket managed simulator (SV1)
# qpu     -> real AWS QPU (expensive, cloud person sets this)

BRAKET_DEVICE_ARN = os.environ.get(
    "BRAKET_DEVICE_ARN",
    "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
)

BRAKET_S3_BUCKET = os.environ.get("BRAKET_S3_BUCKET", "")
BRAKET_S3_PREFIX = os.environ.get("BRAKET_S3_PREFIX", "quantum-test-results")

SHOTS = int(os.environ.get("QUANTUM_SHOTS", "1024"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

CHI_SQUARED_ALPHA = 0.05
FIDELITY_THRESHOLD = 0.90
