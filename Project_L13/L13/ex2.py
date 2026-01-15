import json
import random
from typing import Dict, List, Tuple

STATES = ["A", "C", "G", "T"]

def generate_random_dna(length: int = 50, seed: int | None = None) -> str:
    """Generate a random DNA sequence of given length."""
    if seed is not None:
        random.seed(seed)
    return "".join(random.choice(STATES) for _ in range(length))

def estimate_markov_chain(seq: str, smoothing: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    Estimate a 1st-order Markov chain transition matrix P from a DNA sequence.

    Row-stochastic convention:
      P[from_base][to_base] = P(next = to_base | current = from_base)

    smoothing:
      Add-k (Laplace) smoothing on counts. Use 0.0 for no smoothing.
    """
    # Validate sequence
    bad = [ch for ch in seq if ch not in STATES]
    if bad:
        raise ValueError(f"Sequence contains invalid DNA characters: {sorted(set(bad))}")

    # Initialize counts with smoothing
    counts: Dict[str, Dict[str, float]] = {
        a: {b: float(smoothing) for b in STATES} for a in STATES
    }

    # Count transitions from adjacent symbols
    for curr, nxt in zip(seq, seq[1:]):
        counts[curr][nxt] += 1.0

    # Normalize to probabilities per row
    P: Dict[str, Dict[str, float]] = {}
    for curr in STATES:
        total = sum(counts[curr].values())
        P[curr] = {nxt: counts[curr][nxt] / total for nxt in STATES}

    # Also return integer-ish observed counts (without smoothing) for transparency
    raw_counts: Dict[str, Dict[str, int]] = {a: {b: 0 for b in STATES} for a in STATES}
    for curr, nxt in zip(seq, seq[1:]):
        raw_counts[curr][nxt] += 1

    return P, raw_counts

def save_transition_matrix_json(
    path: str,
    seq: str,
    P: Dict[str, Dict[str, float]],
    raw_counts: Dict[str, Dict[str, int]],
    steps_pred: int = 5
) -> None:
    """
    Save Markov chain transition matrix (and helpful metadata) to a JSON file.
    """
    payload = {
        "model": "first_order_markov_chain",
        "states": STATES,
        "sequence": seq,
        "length": len(seq),
        "transition_matrix": P,        # row-stochastic
        "transition_counts": raw_counts # observed counts (no smoothing)
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    seq = generate_random_dna(length=50, seed=42)  # remove seed for different random output
    P, raw_counts = estimate_markov_chain(seq, smoothing=1.0)  # smoothing=0.0 for raw MLE

    out_file = "dna_transition_matrix.json"
    save_transition_matrix_json(out_file, seq, P, raw_counts)

    print("DNA sequence:", seq)
    print(f"Saved transition matrix JSON to: {out_file}")

if __name__ == "__main__":
    main()
