# simulate_sequence.py

import random
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# Allowed bases
DNA_ALPHABET = ["A", "C", "G", "T"]

# Define a small library of "transposable elements" (motifs)
TE_LIBRARY: Dict[str, str] = {
    "TE1": "ATGCGTACGTTAGCCTAGTA",  # length 20
    "TE2": "TTAACCGGTT",            # length 10
    "TE3": "CGTACGATCG",            # length 10
    "TE4": "AATAAAAA",              # length 8
}

# Colors for plotting each TE type
TE_COLORS: Dict[str, str] = {
    "TE1": "tab:red",
    "TE2": "tab:green",
    "TE3": "tab:blue",
    "TE4": "tab:purple",
}


def random_dna(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    return "".join(random.choice(DNA_ALPHABET) for _ in range(length))


def simulate_sequence(
    min_len: int = 200,
    max_len: int = 400,
    seed: Optional[int] = 42,  # fixed seed for reproducibility
):
    """
    Simulate one DNA sequence and insert 3–4 TE instances.

    - TE1, TE2, TE3 are always inserted.
    - TE4 is optionally inserted (so total is 3 or 4).
    - All TE insertions are constrained to be NON-OVERLAPPING.
    """
    if seed is not None:
        random.seed(seed)

    # Choose sequence length uniformly in [min_len, max_len]
    seq_len = random.randint(min_len, max_len)

    # Original sequence (before TE insertions)
    original_seq = random_dna(seq_len)
    seq = list(original_seq)

    te_instances: List[Dict] = []

    # Decide if we also insert TE4
    insert_te4 = random.choice([False, True])
    te_order = ["TE1", "TE2", "TE3"] + (["TE4"] if insert_te4 else [])

    occupied_intervals: List[Dict] = []  # keep track of placed intervals

    for name in te_order:
        te_seq = TE_LIBRARY[name]
        te_len = len(te_seq)

        # Try to find a non-overlapping place for this TE
        placed = False
        for _ in range(1000):  # plenty of tries
            start = random.randint(0, seq_len - te_len)
            end = start + te_len

            # Check for overlap with all already placed TEs
            overlaps = any(
                not (end <= inst["start_0based"] or inst["end_0based"] <= start)
                for inst in occupied_intervals
            )
            if not overlaps:
                # Place TE here
                seq[start:end] = list(te_seq)
                inst = {
                    "name": name,
                    "start_0based": start,
                    "end_0based": end,
                    "length": te_len,
                }
                te_instances.append(inst)
                occupied_intervals.append(inst)
                placed = True
                break

        if not placed:
            print(f"Warning: could not place {name} without overlap after many tries.")

    # Sort instances by genomic position for nicer output/plot
    te_instances.sort(key=lambda x: x["start_0based"])

    final_seq = "".join(seq)
    return original_seq, final_seq, te_instances


def write_fasta(sequence: str, path: str, header: str = "simulated_sequence") -> None:
    """Write a DNA sequence to a FASTA file."""
    with open(path, "w") as fh:
        fh.write(f">{header}\n")
        # Wrap to 60 chars/line
        for i in range(0, len(sequence), 60):
            fh.write(sequence[i : i + 60] + "\n")


def write_te_library(path: str) -> None:
    """Write TE library as TSV: name\tsequence."""
    with open(path, "w") as fh:
        fh.write("name\tsequence\n")
        for name, seq in TE_LIBRARY.items():
            fh.write(f"{name}\t{seq}\n")


def write_te_truth(instances, path: str) -> None:
    """
    Write TE insertion events as TSV, with 1-based coordinates.

    Columns: name, start_1based, end_1based (end inclusive).
    """
    with open(path, "w") as fh:
        fh.write("name\tstart_1based\tend_1based\n")
        for inst in instances:
            start_1 = inst["start_0based"] + 1
            end_1 = inst["end_0based"]
            fh.write(f'{inst["name"]}\t{start_1}\t{end_1}\n')


def plot_te_events(seq_len: int, instances: List[Dict]) -> None:
    """Use matplotlib to show TE positions as horizontal bars (non-overlapping)."""
    if not instances:
        print("No TE insertions to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 3))

    for idx, inst in enumerate(instances):
        start = inst["start_0based"]
        width = inst["length"]
        color = TE_COLORS.get(inst["name"], "gray")
        ax.broken_barh([(start, width)], (idx - 0.4, 0.8), facecolors=color)
        ax.text(
            start,
            idx,
            inst["name"],
            va="center",
            ha="left",
            fontsize=8,
            color="black",
        )

    ax.set_ylim(-1, len(instances))
    ax.set_xlim(0, seq_len)
    ax.set_xlabel("Position (bp)")
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels([inst["name"] for inst in instances])
    ax.set_title("Simulated transposon insertions (non-overlapping)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    original_seq, final_seq, instances = simulate_sequence()

    seq_len = len(final_seq)

    print(f"Sequence length: {seq_len}")
    print("Simulated TE insertion events (0-based [start, end) ):")
    for inst in instances:
        print(inst)

    # Write files for exercise 2
    write_fasta(final_seq, "simulated_sequence.fasta")
    write_te_library("te_library.tsv")
    write_te_truth(instances, "te_truth.tsv")

    # Show how the sequence changed (truncated to first 120 bp for readability)
    print("\nOriginal sequence (first 120 bp):")
    print(original_seq[:120])
    print("\nFinal sequence with TEs inserted (first 120 bp):")
    print(final_seq[:120])

    # Plot TE positions
    plot_te_events(seq_len, instances)

    print("\nFiles written:")
    print("  simulated_sequence.fasta")
    print("  te_library.tsv")
    print("  te_truth.tsv (ground truth)")
