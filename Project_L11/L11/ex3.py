#!/usr/bin/env python3
"""
Local alignment of influenza A and SARS-CoV-2 genomes with an “in-between layer”.

- Downloads influenza A H1N1 PR8 genome (8 RNA segments) from NCBI (NC_002016–NC_002023)
- Downloads SARS-CoV-2 reference genome NC_045512.2 from NCBI
- Concatenates influenza segments into a single genome string
- Splits both genomes into big windows
- Runs Smith–Waterman local alignment (implemented here) on every window pair
- Builds a window x window similarity matrix
- Visualizes similarities as a heatmap + shows best alignment and 3 similarity scores
"""

import os
import sys
import urllib.request
import urllib.parse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# NCBI download helpers (no Biopython, just urllib)
# ---------------------------------------------------------------------------

INFLUENZA_SEGMENTS = [
    "NC_002016.1",
    "NC_002017.1",
    "NC_002018.1",
    "NC_002019.1",
    "NC_002020.1",
    "NC_002021.1",
    "NC_002022.1",
    "NC_002023.1",
]
SARS_COV_2_ACCESSION = "NC_045512.2"


def download_fasta_from_ncbi(accessions: List[str]) -> str:
    """
    Download one or more nucleotide records from NCBI nuccore as FASTA text.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": ",".join(accessions),
        "rettype": "fasta",
        "retmode": "text",
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    headers = {
        "User-Agent": "LocalAlignmentDemo/1.0 (contact: your_email@example.com)"
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return data.decode("utf-8")


def parse_fasta(text: str) -> Dict[str, str]:
    """
    Very simple FASTA parser: returns dict {header_id: sequence}.
    header_id is the first word after '>'.
    """
    seqs: Dict[str, List[str]] = {}
    current_id = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
            current_id = header.split()[0]
            if current_id not in seqs:
                seqs[current_id] = []
        else:
            if current_id is None:
                raise ValueError("FASTA format error: sequence before header")
            seqs[current_id].append(line.upper())

    return {k: "".join(v) for k, v in seqs.items()}


# ---------------------------------------------------------------------------
# Smith–Waterman local alignment (native implementation)
# ---------------------------------------------------------------------------

def smith_waterman_score(seq1: str,
                         seq2: str,
                         match_score: int = 2,
                         mismatch_score: int = -1,
                         gap_penalty: int = -2) -> int:
    """
    Smith–Waterman: compute only the best local score.
    """
    n = len(seq1)
    m = len(seq2)
    H: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0

    for i in range(1, n + 1):
        ch1 = seq1[i - 1]
        for j in range(1, m + 1):
            ch2 = seq2[j - 1]
            if ch1 == ch2:
                diag = H[i - 1][j - 1] + match_score
            else:
                diag = H[i - 1][j - 1] + mismatch_score
            delete = H[i - 1][j] + gap_penalty
            insert = H[i][j - 1] + gap_penalty
            H[i][j] = max(0, diag, delete, insert)
            if H[i][j] > best:
                best = H[i][j]

    return best


def smith_waterman_alignment(seq1: str,
                             seq2: str,
                             match_score: int = 2,
                             mismatch_score: int = -1,
                             gap_penalty: int = -2) -> Tuple[int, str, str]:
    """
    Smith–Waterman with traceback: returns (best_score, aligned_seq1, aligned_seq2).
    """
    n = len(seq1)
    m = len(seq2)
    H: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    tb: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]  # 0 stop,1 diag,2 up,3 left

    best = 0
    best_pos = (0, 0)

    for i in range(1, n + 1):
        ch1 = seq1[i - 1]
        for j in range(1, m + 1):
            ch2 = seq2[j - 1]
            if ch1 == ch2:
                diag = H[i - 1][j - 1] + match_score
            else:
                diag = H[i - 1][j - 1] + mismatch_score
            up = H[i - 1][j] + gap_penalty
            left = H[i][j - 1] + gap_penalty

            score = 0
            direction = 0
            if diag >= up and diag >= left and diag > 0:
                score = diag
                direction = 1
            elif up >= left and up > 0:
                score = up
                direction = 2
            elif left > 0:
                score = left
                direction = 3

            H[i][j] = score
            tb[i][j] = direction

            if score > best:
                best = score
                best_pos = (i, j)

    aligned1: List[str] = []
    aligned2: List[str] = []
    i, j = best_pos

    while i > 0 and j > 0 and H[i][j] > 0:
        direction = tb[i][j]
        if direction == 1:  # diag
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif direction == 2:  # up
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        elif direction == 3:  # left
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1
        else:
            break

    aligned1.reverse()
    aligned2.reverse()
    return best, "".join(aligned1), "".join(aligned2)


# ---------------------------------------------------------------------------
# Similarity scoring for aligned sequences (3 equations)
# ---------------------------------------------------------------------------

def compute_similarity_scores(aligned1: str, aligned2: str) -> Tuple[float, float, float]:
    """
    Compute 3 different similarity scores for an aligned sequence pair.

    Let:
      M = # matches (same base, not '-')
      X = # mismatches (different bases, neither is '-')
      G = # gap columns (at least one '-')

    Score 1: S1 = M / L                      (fraction identity)
    Score 2: S2 = (M - X) / L                (match–mismatch balance)
    Score 3: S3 = M / (M + X + 2G)           (gap-aware similarity; gaps are weighted double)
    """
    if len(aligned1) != len(aligned2):
        raise ValueError("Aligned sequences must have the same length")

    L = len(aligned1)
    matches = 0
    mismatches = 0
    gaps = 0

    for a, b in zip(aligned1, aligned2):
        if a == "-" or b == "-":
            gaps += 1
        elif a == b:
            matches += 1
        else:
            mismatches += 1

    if L == 0:
        return 0.0, 0.0, 0.0

    # Score 1: percent identity (fraction)
    s1 = matches / L

    # Score 2: match–mismatch balance
    s2 = (matches - mismatches) / L

    # Score 3: gap-aware similarity, gaps counted twice
    denom = matches + mismatches + 2 * gaps
    s3 = matches / denom if denom > 0 else 0.0

    return s1, s2, s3


# ---------------------------------------------------------------------------
# In-between layer: windowed comparison
# ---------------------------------------------------------------------------

def make_windows(seq: str, window_size: int) -> List[Tuple[int, str]]:
    """
    Split a sequence into non-overlapping windows of given size.
    Returns list of (start_index, subsequence).
    """
    windows: List[Tuple[int, str]] = []
    for start in range(0, len(seq), window_size):
        end = min(start + window_size, len(seq))
        windows.append((start, seq[start:end]))
    return windows


def build_similarity_matrix(
    seq1: str,
    seq2: str,
    window_size: int,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> Tuple[List[List[float]], List[Tuple[int, str]], List[Tuple[int, str]], Tuple[int, int]]:
    """
    Build a window x window similarity matrix using Smith–Waterman scores.
    """
    windows1 = make_windows(seq1, window_size)
    windows2 = make_windows(seq2, window_size)

    n = len(windows1)
    m = len(windows2)
    similarities: List[List[float]] = [[0.0] * m for _ in range(n)]

    max_score_seen = 0
    best_pair = (0, 0)

    for i, (start1, w1) in enumerate(windows1):
        print(f"Computing row {i+1}/{n} of window similarity matrix...")
        for j, (start2, w2) in enumerate(windows2):
            score = smith_waterman_score(
                w1, w2,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_penalty=gap_penalty,
            )
            similarities[i][j] = float(score)
            if score > max_score_seen:
                max_score_seen = score
                best_pair = (i, j)

    # Normalize by theoretical max
    for i, (_, w1) in enumerate(windows1):
        for j, (_, w2) in enumerate(windows2):
            max_possible = match_score * min(len(w1), len(w2))
            if max_possible > 0:
                similarities[i][j] /= max_possible

    return similarities, windows1, windows2, best_pair


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Parameters
    window_size = 1000
    match_score = 2
    mismatch_score = -1
    gap_penalty = -2

    flu_fasta_file = "influenza_PR8_segments.fasta"
    covid_fasta_file = "sars_cov_2_NC_045512.2.fasta"

    if not os.path.exists(flu_fasta_file):
        print("Downloading influenza A PR8 segments from NCBI...")
        flu_fasta_text = download_fasta_from_ncbi(INFLUENZA_SEGMENTS)
        with open(flu_fasta_file, "w") as f:
            f.write(flu_fasta_text)
    else:
        with open(flu_fasta_file) as f:
            flu_fasta_text = f.read()

    if not os.path.exists(covid_fasta_file):
        print("Downloading SARS-CoV-2 reference genome from NCBI...")
        covid_fasta_text = download_fasta_from_ncbi([SARS_COV_2_ACCESSION])
        with open(covid_fasta_file, "w") as f:
            f.write(covid_fasta_text)
    else:
        with open(covid_fasta_file) as f:
            covid_fasta_text = f.read()

    flu_sequences = parse_fasta(flu_fasta_text)
    covid_sequences = parse_fasta(covid_fasta_text)

    influenza_genome_parts: List[str] = []
    for acc in INFLUENZA_SEGMENTS:
        if acc in flu_sequences:
            influenza_genome_parts.append(flu_sequences[acc])
        else:
            print(f"Warning: segment {acc} not found in influenza FASTA.")
    influenza_genome = "".join(influenza_genome_parts)

    if not covid_sequences:
        print("Error: no SARS-CoV-2 sequence parsed")
        sys.exit(1)
    covid_genome = next(iter(covid_sequences.values()))

    print(f"Influenza genome length (concatenated): {len(influenza_genome)}")
    print(f"SARS-CoV-2 genome length: {len(covid_genome)}")

    print("\nBuilding window similarity matrix using Smith–Waterman...")
    sim_matrix, flu_windows, covid_windows, best_pair = build_similarity_matrix(
        influenza_genome,
        covid_genome,
        window_size=window_size,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_penalty=gap_penalty,
    )

    best_i, best_j = best_pair
    best_flu_start, best_flu_seq = flu_windows[best_i]
    best_covid_start, best_covid_seq = covid_windows[best_j]

    print(f"\nBest window pair: influenza window {best_i}, SARS-CoV-2 window {best_j}")
    print(f"Influenza window start: {best_flu_start}")
    print(f"SARS-CoV-2 window start: {best_covid_start}")

    print("\nRunning Smith–Waterman on best window pair for detailed alignment...")
    best_score, aligned_flu, aligned_covid = smith_waterman_alignment(
        best_flu_seq,
        best_covid_seq,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_penalty=gap_penalty,
    )

    # --- NEW: compute similarity scores on the aligned sequences ---
    s1, s2, s3 = compute_similarity_scores(aligned_flu, aligned_covid)
    pid_percent = s1 * 100.0

    print(f"\nBest local alignment score (Smith–Waterman): {best_score}")
    print(f"Score 1 (percent identity): {pid_percent:.2f} %")
    print(f"Score 2 (match–mismatch balance): {s2:.4f}")
    print(f"Score 3 (gap-aware similarity): {s3:.4f}")

    # Build match line (for plotting)
    match_line_chars: List[str] = []
    for a, b in zip(aligned_flu, aligned_covid):
        if a == b and a != "-":
            match_line_chars.append("|")
        else:
            match_line_chars.append(" ")
    match_line = "".join(match_line_chars)

    # ------------------------------------------------------------------
    # Visualization: heatmap + alignment in the same figure
    # ------------------------------------------------------------------
    print("Drawing similarity heatmap and alignment...")

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # Top: heatmap
    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(
        sim_matrix,
        origin="lower",
        aspect="auto",
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Normalized Smith–Waterman score")

    ax_heat.set_xlabel("SARS-CoV-2 genome window index")
    ax_heat.set_ylabel("Influenza A genome window index")
    ax_heat.set_title(
        "Local similarity between influenza A (PR8) and SARS-CoV-2 genomes\n"
        f"window size = {window_size} bp"
    )

    # Bottom: alignment text
    ax_align = fig.add_subplot(gs[1, 0])
    ax_align.set_axis_off()
    ax_align.set_xlim(0, 1)
    ax_align.set_ylim(0, 1)

    # If alignment is very long, show only first N characters
    max_chars = 140
    if len(aligned_flu) > max_chars:
        aligned_flu_plot = aligned_flu[:max_chars] + "..."
        aligned_covid_plot = aligned_covid[:max_chars] + "..."
        match_line_plot = match_line[:max_chars] + "..."
    else:
        aligned_flu_plot = aligned_flu
        aligned_covid_plot = aligned_covid
        match_line_plot = match_line

    text_y_top = 0.75
    text_y_mid = 0.5
    text_y_bot = 0.25

    ax_align.text(
        0.01,
        text_y_top,
        "Influenza : ",
        fontweight="bold",
        family="monospace",
        transform=ax_align.transAxes,
        va="center",
    )
    ax_align.text(
        0.20,
        text_y_top,
        aligned_flu_plot,
        family="monospace",
        transform=ax_align.transAxes,
        va="center",
    )

    ax_align.text(
        0.20,
        text_y_mid,
        match_line_plot,
        family="monospace",
        transform=ax_align.transAxes,
        va="center",
    )

    ax_align.text(
        0.01,
        text_y_bot,
        "SARS-CoV-2:",
        fontweight="bold",
        family="monospace",
        transform=ax_align.transAxes,
        va="center",
    )
    ax_align.text(
        0.20,
        text_y_bot,
        aligned_covid_plot,
        family="monospace",
        transform=ax_align.transAxes,
        va="center",
    )

    ax_align.set_title(
        "Best local alignment between genomes "
        f"(windows {best_i} / {best_j}, SW score={best_score}, "
        f"PID={pid_percent:.1f} %, S2={s2:.3f}, S3={s3:.3f})",
        pad=20,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
