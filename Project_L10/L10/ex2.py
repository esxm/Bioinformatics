

import os
import math
import csv


# ---------------------- FASTA READER ---------------------- #

def read_fasta(filepath):
    """
    Simple FASTA parser.

    Returns:
        list of (header, sequence) tuples
    """
    sequences = []
    header = None
    seq_chunks = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous sequence, if any
                if header is not None:
                    sequences.append((header, "".join(seq_chunks).upper()))
                header = line[1:].strip()  # without '>'
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # Save last one
        if header is not None:
            sequences.append((header, "".join(seq_chunks).upper()))

    return sequences


# ---------------------- IC & C+G% ---------------------- #

def shannon_entropy(window):
    """
    Shannon entropy H (base 2) for nucleotides in the window.
    """
    length = len(window)
    if length == 0:
        return 0.0

    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for base in window:
        if base in counts:
            counts[base] += 1

    H = 0.0
    for base in "ACGT":
        p = counts[base] / length
        if p > 0.0:
            H -= p * math.log2(p)
    return H


def information_content(window):
    """
    IC = 2 - H (with log2; maximum for 4 symbols is log2(4) = 2)
    """
    H = shannon_entropy(window)
    return 2.0 - H


def cg_fraction(window):
    """
    Fraction of C+G in the window, between 0 and 1.
    """
    length = len(window)
    if length == 0:
        return 0.0
    c = window.count("C")
    g = window.count("G")
    return (c + g) / length


# ---------------------- SLIDING WINDOW ---------------------- #

def sliding_window_ods(seq, window_size=100, step=1):
    """
    Slide a window along 'seq' and compute (start, end, CG%, IC) for each window.

    Returns:
        list of tuples: (start, end, cg_percent, ic)
        where start/end are 0-based indices (end exclusive).
    """
    results = []
    n = len(seq)
    if n < window_size:
        return results

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = seq[start:end]
        cg = cg_fraction(window) * 100.0       # percentage
        ic = information_content(window)       # bits
        results.append((start, end, cg, ic))

    return results


# ---------------------- SAVE ODS TO FILE ---------------------- #

def save_ods_table(ods_data, promoter_idx, promoter_header, out_dir):
    """
    Save ODS data (list of tuples) to a CSV file in out_dir.

    Each row: window_start, window_end, cg_percent, ic_bits
    """
    # Safe filename: remove weird chars
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in promoter_header)
    filename = f"ODS_{promoter_idx:03d}_{safe_id}.csv"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["window_start", "window_end", "cg_percent", "ic_bits"])
        for (start, end, cg, ic) in ods_data:
            writer.writerow([start, end, f"{cg:.6f}", f"{ic:.6f}"])

    return out_path


# ---------------------- MAIN PIPELINE ---------------------- #

def main():
    # ---- CONFIG ----
    # Adjust these if needed:
    FASTA_FILE = "promoters_list"  # or "promoters_list.fa"
    WINDOW_SIZE = 100
    STEP = 1

    # Try a couple of possible file names
    candidate_paths = [
        FASTA_FILE,
        FASTA_FILE + ".fa",
        FASTA_FILE + ".fasta",
        FASTA_FILE + ".txt",
    ]
    fasta_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            fasta_path = p
            break

    if fasta_path is None:
        raise FileNotFoundError(
            "Could not find promoters_list file. "
            "Tried: 'promoters_list', 'promoters_list.fa', "
            "'promoters_list.fasta', 'promoters_list.txt'."
        )

    # Output folder for ODS data
    out_dir = "ODS"
    os.makedirs(out_dir, exist_ok=True)

    # Read promoters
    promoters = read_fasta(fasta_path)
    print(f"Loaded {len(promoters)} promoters from '{fasta_path}'")

    # For each promoter, compute ODS and save table
    for idx, (header, seq) in enumerate(promoters, start=1):
        ods_data = sliding_window_ods(seq, WINDOW_SIZE, STEP)

        if not ods_data:
            print(f"[Warning] Promoter {idx} ('{header}') is shorter than "
                  f"the window size ({WINDOW_SIZE}). Skipping.")
            continue

        out_path = save_ods_table(ods_data, idx, header, out_dir)
        print(f"Saved ODS data for promoter {idx} to '{out_path}'")

    print("Done. All ODS data files are in the 'ODS' folder.")


if __name__ == "__main__":
    main()
