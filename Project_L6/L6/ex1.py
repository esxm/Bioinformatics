#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gel electrophoresis simulation:
- Load one DNA sequence (FASTA)
- Verify total length in [1000, 3000] nt
- Take 10 random samples (each 100–3000 bp; capped by total length)
- Plot a gel lane showing the 10 samples (shorter fragments run farther)
- (Optional) Compare with a 5-enzyme restriction digest lane

Author: you :)
"""

import math
import random
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors

# --- Visual style ---
GEL_THEME = "dark"          # "dark" or "light"
BAND_CMAP = "viridis"       # any matplotlib cmap name (e.g., "magma", "plasma", "turbo")
BAND_ALPHA = 0.95           # 0..1 transparency for bands


# =========================
# ---- Configuration ------
# =========================
FASTA_PATH = r"Boebeisivirus.fasta"

# Seed for reproducibility (set to an integer to get repeatable results)
RANDOM_SEED = None          # e.g., 42  (or None for true randomness)

# If not None, save the gel figure(s) as PNG to this path (e.g., "gel.png")
SAVE_PNG = None             # e.g., "random_samples_gel.png"

# Assignment parameters
VERIFY_MIN_TOTAL = 1000     # total FASTA length must be >= this
VERIFY_MAX_TOTAL = 3000     # total FASTA length must be <= this
N_SAMPLES = 10              # number of random samples
SAMPLE_MIN_LEN = 100        # each random sample min length
SAMPLE_MAX_LEN = 3000       # each random sample max length (will be capped by total length)

# Optional: also do a 5-enzyme restriction digest for comparison
SHOW_DIGEST_COMPARISON = False

# =========================
# ---- Enzyme Database ----
# =========================
ENZYMES = {
    "EcoRI":  {"site": "GAATTC",   "cut_index": 1},  # G^AATTC
    "BamHI":  {"site": "GGATCC",   "cut_index": 1},  # G^GATCC
    "HindIII":{"site": "AAGCTT",   "cut_index": 1},  # A^AGCTT
    "NotI":   {"site": "GCGGCCGC", "cut_index": 2},  # GC^GGCCGC
    "XhoI":   {"site": "CTCGAG",   "cut_index": 1},  # C^TCGAG
    "PstI":   {"site": "CTGCAG",   "cut_index": 5},  # CTGCA^G
    "SacI":   {"site": "GAGCTC",   "cut_index": 5},  # GAGCT^C
    "KpnI":   {"site": "GGTACC",   "cut_index": 5},  # GGTAC^C
    "NcoI":   {"site": "CCATGG",   "cut_index": 1},  # C^CATGG
    "NheI":   {"site": "GCTAGC",   "cut_index": 1},  # G^CTAGC
    "SpeI":   {"site": "ACTAGT",   "cut_index": 1},  # A^CTAGT
    "XbaI":   {"site": "TCTAGA",   "cut_index": 1},  # T^CTAGA
}

# =========================
# ---- I/O & Parsing ------
# =========================
def read_fasta_one_sequence(path):
    """
    Read the FIRST sequence from a FASTA file, uppercase it, and strip
    non-ACGT characters. Returns (header, sequence).
    """
    header = "sequence"
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chunks:
                    break  # first sequence done
                header = line[1:].strip() or "sequence"
            else:
                seq_chunks.append(line.upper())
    seq = "".join(seq_chunks)
    seq = "".join(ch for ch in seq if ch in "ACGT")
    if not seq:
        raise ValueError(f"No A/C/G/T bases found in FASTA: {path}")
    return header, seq

# =========================
# ---- Random Sampling ----
# =========================
def verify_fasta_requirements(seq, min_total=1000, max_total=3000):
    """Return (ok, message). ok=True if length is within required bounds."""
    n = len(seq)
    if n < min_total:
        return False, f"Sequence is too short: {n} nt (< {min_total})."
    if n > max_total:
        return False, f"Sequence is too long: {n} nt (> {max_total})."
    return True, f"Sequence length OK: {n} nt (within {min_total}-{max_total})."

def take_random_samples(seq, n=10, min_len=100, max_len=3000, rng=None):
    """
    Return list of tuples: (start, end, length, subseq) for n random substrings.
    max_len is capped by total sequence length.
    """
    if rng is None:
        rng = random
    L = len(seq)
    max_len = min(max_len, L)
    if max_len < min_len:
        raise ValueError(f"Sequence ({L} nt) is shorter than min_len={min_len}.")
    samples = []
    for _ in range(n):
        k = rng.randint(min_len, max_len)  # inclusive
        start = rng.randint(0, L - k)      # inclusive
        end = start + k
        samples.append((start, end, k, seq[start:end]))
    return samples

# =========================
# ---- Digest Utilities ---
# =========================
def find_sites(seq, motif):
    sites = []
    m = len(motif)
    i = seq.find(motif, 0)
    while i != -1:
        sites.append(i)
        i = seq.find(motif, i + 1)
    return sites

def enzyme_cut_positions(seq, site, cut_index):
    positions = []
    for start in find_sites(seq, site):
        positions.append(start + cut_index)
    return positions

def pick_five_enzymes(seq, db, n=5):
    cutters = []
    non_cutters = []
    for name, info in db.items():
        cuts = enzyme_cut_positions(seq, info["site"], info["cut_index"])
        (cutters if cuts else non_cutters).append(name)
    random.shuffle(cutters)
    random.shuffle(non_cutters)
    chosen = cutters[:n]
    if len(chosen) < n:
        chosen += non_cutters[: (n - len(chosen))]
    return chosen

def digest_sequence(seq, enzymes, db):
    all_cuts = set()
    per_enzyme = {}
    for name in enzymes:
        site = db[name]["site"]
        cut_index = db[name]["cut_index"]
        positions = enzyme_cut_positions(seq, site, cut_index)
        per_enzyme[name] = sorted(positions)
        all_cuts.update(positions)

    cut_positions = sorted(all_cuts)
    coords = [0] + cut_positions + [len(seq)]
    frags = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
    return sorted(frags, reverse=True), per_enzyme, cut_positions

# =========================
# ---- Gel Plotting -------
# =========================
def migration_positions_bp(lengths):
    """
    Map fragment lengths (bp) to y positions (0..1) using an inverse log scale:
    shorter -> larger migration -> lower y.
    """
    logs = [math.log10(L) for L in lengths]
    min_log, max_log = min(logs), max(logs)
    if max_log == min_log:
        return [0.5] * len(lengths)
    top_margin, bottom_margin = 0.1, 0.9
    usable = bottom_margin - top_margin
    return [top_margin + ((max_log - lg) / (max_log - min_log)) * usable for lg in logs]

def _draw_lane(ax, band_lengths, lane_center_x=0.5, lane_width=0.28, label="Lane"):
    """Internal helper: draw one lane worth of bands and lane box with colors."""
    # Theme colors
    dark = (GEL_THEME.lower() == "dark")
    gel_edge = "white" if dark else "black"
    text_color = "white" if dark else "black"
    gel_bg = (0.06, 0.06, 0.08, 1.0) if dark else (0.96, 0.96, 0.98, 1.0)

    # Lane geometry
    x_left = lane_center_x - lane_width / 2
    x_right = lane_center_x + lane_width / 2

    # Fill the lane background
    lane_rect = Rectangle((x_left, 0.08), lane_width, 0.84,
                          fill=True, facecolor=gel_bg, edgecolor=gel_edge, linewidth=1.0)
    ax.add_patch(lane_rect)

    # Positions and colormap mapping (shorter -> farther, also brighter by default)
    positions = migration_positions_bp(band_lengths)
    vmin, vmax = max(1, min(band_lengths)), max(band_lengths)
    # Use a log scale for color mapping to match migration logic
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(BAND_CMAP)

    # Draw bands
    for y, L in sorted(zip(positions, band_lengths), key=lambda t: t[0]):
        # replace color line inside the loop with this:
        color = cmap(norm(vmax - (L - vmin) + vmin))  # simple invert
        ax.plot([x_left + 0.01, x_right - 0.01], [y, y],
                linewidth=4.2, color=color, alpha=BAND_ALPHA, solid_capstyle="round")
        # Length label
        ax.text(x_right + 0.03, y, f"{L} bp", va="center", fontsize=9, color=text_color)

    # Lane label
    ax.text((x_left + x_right) / 2, 0.03, label, ha="center", va="center",
            fontsize=10, color=text_color)

def plot_single_lane_gel(fragment_lengths, lane_label="Digest", title=None, save_path=None):
    """Plot a single lane (used by the digest view)."""
    dark = (GEL_THEME.lower() == "dark")
    frame_edge = "white" if dark else "black"
    title_color = "white" if dark else "black"

    fig, ax = plt.subplots(figsize=(4.5, 6.5))
    # Figure/axes background
    fig.patch.set_facecolor((0.02, 0.02, 0.03, 1.0) if dark else "white")
    ax.set_facecolor((0.02, 0.02, 0.03, 1.0) if dark else "white")

    # Gel frame
    ax.add_patch(Rectangle((0.18, 0.05), 0.64, 0.9, fill=False, linewidth=1.5, edgecolor=frame_edge))

    # Lane
    _draw_lane(ax, fragment_lengths, lane_center_x=0.5, lane_width=0.28, label=lane_label)

    if title:
        ax.set_title(title, fontsize=11, pad=10, color=title_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def plot_random_samples_gel(sample_lengths, lane_label="10 random samples", title=None, save_path=None):
    """Plot a single lane with the random samples bands."""
    plot_single_lane_gel(sample_lengths, lane_label=lane_label, title=title, save_path=save_path)

# =========================
# --------- Main ----------
# =========================
if __name__ == "__main__":
    # Seed reproducibility
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    else:
        random.seed()

    fasta_path = pathlib.Path(FASTA_PATH)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    header, seq = read_fasta_one_sequence(fasta_path)

    # 1) Verify FASTA meets assignment requirement
    ok, msg = verify_fasta_requirements(seq, VERIFY_MIN_TOTAL, VERIFY_MAX_TOTAL)
    print(f"Sequence: {header}")
    print(msg)
    if not ok:
        raise SystemExit("FASTA does not meet the required length; aborting per assignment.")

    # 2) Take 10 random samples
    samples = take_random_samples(seq,
                                  n=N_SAMPLES,
                                  min_len=SAMPLE_MIN_LEN,
                                  max_len=SAMPLE_MAX_LEN)
    # Store samples (full tuples) and also collect lengths for plotting
    sample_sequences = [sub for (_s, _e, _k, sub) in samples]
    sample_lengths = [k for (_s, _e, k, _sub) in samples]

    print("\nRandom samples (start..end, length):")
    for i, (s, e, k, _sub) in enumerate(samples, start=1):
        print(f"  {i:>2}: {s}..{e} (len={k} bp)")

    print("\nRandom sample lengths (bp):", sample_lengths)

    # 3) Plot the random-samples gel lane
    plot_random_samples_gel(
        sample_lengths,
        lane_label=f"{N_SAMPLES} random samples",
        title=f"Random fragments from {header}",
        save_path=SAVE_PNG
    )

    # 4) (Optional) Compare with a 5-enzyme digest lane
    if SHOW_DIGEST_COMPARISON:
        chosen = pick_five_enzymes(seq, ENZYMES, n=5)
        fragments, per_enzyme_sites, all_cut_positions = digest_sequence(seq, chosen, ENZYMES)

        print("\n--- 5-Enzyme Digest Summary ---")
        print("Chosen enzymes (5):", ", ".join(chosen))
        for name in chosen:
            site = ENZYMES[name]["site"]
            sites = per_enzyme_sites[name]
            print(f"  - {name} ({site}) -> {len(sites)} site(s): {sites if sites else '—'}")
        print(f"Total unique cut positions: {len(all_cut_positions)} -> fragments: {len(fragments)}")
        print("Fragment sizes (bp, largest first):", fragments)

        plot_single_lane_gel(
            fragment_lengths=fragments,
            lane_label="5-enzyme digest",
            title=f"Digest of {header}"
        )
