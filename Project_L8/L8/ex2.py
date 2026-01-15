from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Hardcoded input file paths (matching simulate_sequence.py output)
SEQUENCE_FASTA_PATH = "simulated_sequence.fasta"
TE_LIBRARY_PATH = "te_library.tsv"

# ANSI color codes for terminal highlighting (optional)
ANSI_RESET = "\033[0m"
TE_ANSI_COLORS: Dict[str, str] = {
    "TE1": "\033[91m",  # red
    "TE2": "\033[92m",  # green
    "TE3": "\033[94m",  # blue
    "TE4": "\033[95m",  # magenta
}

# Colors for plotting (matplotlib)
TE_COLORS: Dict[str, str] = {
    "TE1": "tab:red",
    "TE2": "tab:green",
    "TE3": "tab:blue",
    "TE4": "tab:purple",
}


# ---------- I/O helpers ----------

def read_fasta(path: str) -> str:
    """Read the first sequence from a FASTA file and return it as a single string."""
    seq_lines: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Header line, skip
                continue
            seq_lines.append(line.upper())
    return "".join(seq_lines)


def read_te_library(path: str) -> Dict[str, str]:
    """
    Read TE library from TSV file: name \t sequence.
    Returns a dict: {TE_name: TE_sequence}.
    """
    tes: Dict[str, str] = {}
    with open(path) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, seq = line.split("\t")
            tes[name] = seq.upper()
    return tes


# ---------- Detection ----------

def find_all(pattern: str, text: str) -> List[int]:
    """
    Return all starting indices (0-based) of pattern in text, including overlapping hits.
    """
    starts: List[int] = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        starts.append(idx)
        # Move only one base forward => overlapping matches allowed
        start = idx + 1
    return starts


def scan_for_tes(sequence: str, te_library: Dict[str, str]) -> List[Tuple[str, int, int]]:
    """
    Scan the sequence for all occurrences of each TE motif.

    Returns a list of tuples: (TE_name, start_1based, end_1based),
    where coordinates are 1-based and end is inclusive.
    """
    hits: List[Tuple[str, int, int]] = []

    for name, motif in te_library.items():  # iteration order: TE1, TE2, TE3, TE4
        if not motif:
            continue
        for start0 in find_all(motif, sequence):
            end0 = start0 + len(motif)  # Python end index (exclusive)
            start1 = start0 + 1         # convert to 1-based
            end1 = end0                 # end inclusive
            hits.append((name, start1, end1))

    # IMPORTANT: don't resort by position here – we want hits in TE-name order.
    return hits


# ---------- Terminal coloring (optional) ----------

def build_color_map_terminal(
    seq_len: int,
    hits: List[Tuple[str, int, int]],
) -> List[Optional[str]]:
    """
    For each position in the sequence, decide which ANSI color (if any) to apply.
    """
    color_for_pos: List[Optional[str]] = [None] * seq_len

    for name, start1, end1 in hits:
        color = TE_ANSI_COLORS.get(name)
        if color is None:
            continue
        start0 = start1 - 1
        end0 = end1
        for i in range(start0, end0):
            if 0 <= i < seq_len:
                color_for_pos[i] = color

    return color_for_pos


def pretty_print_colored_sequence(
    sequence: str,
    color_for_pos: List[Optional[str]],
    line_width: int = 60,
) -> None:
    """
    Print the DNA sequence with colored bases where TEs are present (terminal output).
    """
    seq_len = len(sequence)
    print("\nColored sequence (bases in detected TEs are highlighted):")

    for i in range(0, seq_len, line_width):
        segment = sequence[i : i + line_width]
        colored_line_chars: List[str] = []

        for j, base in enumerate(segment):
            pos = i + j
            color = color_for_pos[pos]
            if color is not None:
                colored_line_chars.append(f"{color}{base}{ANSI_RESET}")
            else:
                colored_line_chars.append(base)

        print("".join(colored_line_chars))


def print_legend_terminal(te_library: Dict[str, str]) -> None:
    """
    Print a small legend showing which ANSI color corresponds to which TE.
    """
    print("\nLegend (terminal TE colors):")
    for name in te_library.keys():
        color = TE_ANSI_COLORS.get(name, "")
        if color:
            print(f"  {color}{name}{ANSI_RESET}: motif length {len(te_library[name])}")
        else:
            print(f"  {name}: motif length {len(te_library[name])} (no color configured)")


# ---------- Plotting (matplotlib) ----------

def plot_detected_tes(seq_len: int, instances: List[Dict]) -> None:
    """Use matplotlib to show detected TE positions as horizontal bars."""
    if not instances:
        print("\nNo TEs to plot.")
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
    ax.set_title("Detected transposon hits (in TE order)")
    plt.tight_layout()
    plt.show()


# ---------- Main ----------

def main():
    seq = read_fasta(SEQUENCE_FASTA_PATH)
    tes = read_te_library(TE_LIBRARY_PATH)

    print(f"Loaded sequence of length {len(seq)} from {SEQUENCE_FASTA_PATH}")
    print(f"Loaded {len(tes)} TE motifs from {TE_LIBRARY_PATH}")
    print()

    raw_hits = scan_for_tes(seq, tes)

    if not raw_hits:
        print("No transposable elements found (with these exact motifs).")
        return

    # Reorder hits strictly by TE order as in the library (TE1, TE2, TE3, TE4)
    ordered_hits: List[Tuple[str, int, int]] = []
    for te_name in tes.keys():
        these = [h for h in raw_hits if h[0] == te_name]
        # In case there are >1 hits for same TE, sort them by position
        these.sort(key=lambda x: x[1])
        ordered_hits.extend(these)

    print("Found TE occurrences (1-based coordinates, end inclusive):")
    print("TE_name\tstart\tend")
    for name, start, end in ordered_hits:
        print(f"{name}\t{start}\t{end}")

    # ----- Colored sequence in terminal -----
    color_for_pos = build_color_map_terminal(len(seq), ordered_hits)
    print_legend_terminal(tes)
    pretty_print_colored_sequence(seq, color_for_pos)

    # ----- Prepare instances for plotting -----
    instances: List[Dict] = []
    for name, start1, end1 in ordered_hits:
        instances.append(
            {
                "name": name,
                "start_1based": start1,
                "end_1based": end1,
                "start_0based": start1 - 1,
                "end_0based": end1,
                "length": end1 - start1 + 1,
            }
        )

    plot_detected_tes(len(seq), instances)


if __name__ == "__main__":
    main()
