import math
from scipy.stats import chisquare
import numpy as np
import config


def normalize_counts(counts, shots=None):
    """Convert raw counts dict to probability distribution dict."""
    total = shots or sum(counts.values())
    return {state: count / total for state, count in counts.items()}


def align_distributions(dist_a, dist_b):
    """
    Ensure both distributions share the same key set.
    Missing keys get probability 0.
    """
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    aligned_a = {k: dist_a.get(k, 0.0) for k in all_keys}
    aligned_b = {k: dist_b.get(k, 0.0) for k in all_keys}
    return aligned_a, aligned_b


def chi_squared_test(observed_counts, expected_probs, shots):
    """
    Chi-squared goodness-of-fit test.
    observed_counts: dict of bitstring -> count
    expected_probs:  dict of bitstring -> expected probability
    Returns dict with statistic, p_value, and pass/fail.
    """
    all_keys = set(observed_counts.keys()) | set(expected_probs.keys())

    observed = np.array([observed_counts.get(k, 0) for k in all_keys], dtype=float)
    expected = np.array([expected_probs.get(k, 0.0) * shots for k in all_keys], dtype=float)

    # avoid division by zero: merge bins with expected < 5
    mask = expected >= 5
    if mask.sum() < 2:
        return {
            "statistic": None,
            "p_value": None,
            "passed": None,
            "note": "insufficient expected counts for chi-squared test",
        }

    stat, p_value = chisquare(observed[mask], f_exp=expected[mask])
    passed = bool(p_value > config.CHI_SQUARED_ALPHA)
    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "alpha": config.CHI_SQUARED_ALPHA,
        "passed": passed,
    }


def state_fidelity(dist_a, dist_b):
    """
    Classical fidelity (Bhattacharyya coefficient) between two distributions.
    F = sum_x sqrt(p(x) * q(x))
    Returns value in [0, 1], where 1 is identical distributions.
    """
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    fidelity = sum(
        math.sqrt(dist_a.get(k, 0.0) * dist_b.get(k, 0.0))
        for k in all_keys
    )
    return round(fidelity, 6)


def top_state_accuracy(counts, expected_state):
    """
    Checks whether the most frequent measurement outcome matches expected_state.
    Returns dict with top_state, expected, match, and probability of top state.
    """
    total = sum(counts.values())
    top_state = max(counts, key=counts.get)
    top_prob = counts[top_state] / total
    return {
        "top_state": top_state,
        "expected_state": expected_state,
        "match": top_state == expected_state,
        "top_state_probability": round(top_prob, 6),
    }


def distribution_entropy(dist):
    """Shannon entropy of a probability distribution."""
    entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    return round(entropy, 6)


def validate_grover(counts, target_state, n_qubits, shots):
    """Full validation suite for Grover's algorithm results."""
    n_states = 2 ** n_qubits
    expected_state_str = format(target_state, f"0{n_qubits}b")

    # theoretical Grover amplification probability
    n_iter = max(1, math.floor(math.pi / 4 * math.sqrt(n_states)))
    theta = math.asin(1 / math.sqrt(n_states))
    theoretical_prob = math.sin((2 * n_iter + 1) * theta) ** 2
    other_prob = (1 - theoretical_prob) / (n_states - 1)

    expected_probs = {
        format(i, f"0{n_qubits}b"): (theoretical_prob if i == target_state else other_prob)
        for i in range(n_states)
    }

    dist = normalize_counts(counts, shots)
    chi = chi_squared_test(counts, expected_probs, shots)
    accuracy = top_state_accuracy(counts, expected_state_str)
    entropy = distribution_entropy(dist)

    return {
        "algorithm": "grover",
        "chi_squared": chi,
        "top_state_accuracy": accuracy,
        "entropy": entropy,
        "theoretical_target_probability": round(theoretical_prob, 6),
        "observed_target_probability": round(dist.get(expected_state_str, 0.0), 6),
    }


def validate_qft(counts, n_qubits, shots):
    """
    QFT validation: checks that output is non-trivial (not all in one state)
    and that entropy is within expected range for a transformed state.
    """
    dist = normalize_counts(counts, shots)
    entropy = distribution_entropy(dist)
    n_states = 2 ** n_qubits
    max_entropy = math.log2(n_states)

    top_state = max(counts, key=counts.get)
    top_prob = counts[top_state] / sum(counts.values())

    # QFT output should not collapse to a single state (entropy near 0)
    # and should not be a no-op (all amplitude stays in input state).
    # Near-uniform spread is valid for many structured inputs.
    top_state_dominant = top_prob > 0.95
    non_trivial = entropy > 0.1 * max_entropy and not top_state_dominant

    return {
        "algorithm": "qft",
        "entropy": entropy,
        "max_possible_entropy": round(max_entropy, 6),
        "entropy_ratio": round(entropy / max_entropy, 6),
        "non_trivial_output": non_trivial,
        "top_state": top_state,
        "top_state_probability": round(top_prob, 6),
        "passed": non_trivial,
    }


def validate_vqe(vqe_results):
    """
    VQE validation: checks energy variance across iterations
    and that at least one iteration produces a negative energy estimate.
    """
    energies = [r["energy"] for r in vqe_results]
    mean_energy = float(np.mean(energies))
    std_energy = float(np.std(energies))
    min_energy = float(np.min(energies))
    has_negative = min_energy < 0

    return {
        "algorithm": "vqe",
        "iterations": len(vqe_results),
        "mean_energy": round(mean_energy, 6),
        "std_energy": round(std_energy, 6),
        "min_energy": round(min_energy, 6),
        "has_negative_energy_estimate": has_negative,
        "passed": has_negative,
    }


def compare_simulator_vs_hardware(sim_counts, hw_counts, shots):
    """
    Compares distributions from simulator and hardware runs.
    Returns fidelity and per-state probability differences.
    """
    sim_dist = normalize_counts(sim_counts, shots)
    hw_dist = normalize_counts(hw_counts, shots)
    fidelity = state_fidelity(sim_dist, hw_dist)

    sim_aligned, hw_aligned = align_distributions(sim_dist, hw_dist)
    per_state_diff = {
        state: round(abs(sim_aligned[state] - hw_aligned[state]), 6)
        for state in sim_aligned
    }

    passed = fidelity >= config.FIDELITY_THRESHOLD

    return {
        "fidelity": fidelity,
        "fidelity_threshold": config.FIDELITY_THRESHOLD,
        "passed": passed,
        "per_state_absolute_difference": per_state_diff,
        "mean_absolute_error": round(float(np.mean(list(per_state_diff.values()))), 6),
    }
