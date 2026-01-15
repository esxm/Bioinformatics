#!/usr/bin/env python3
"""
Sliding-window Tm with hardcoded thresholds and two charts per sequence.

Chart 1 (signals_with_thresholds):
  - Tm_simple (blue, solid) + its horizontal threshold (blue, dashed)
  - Tm_complex (orange, solid) + its horizontal threshold (orange, dashed)

Chart 2 (regions_above_threshold):
  - Rectangles per window showing ONLY the parts above the threshold
    (top strip = P1 / Tm_simple in blue, bottom strip = P2 / Tm_complex in orange)

Also writes a CSV with the window metrics and a summary CSV with areas above threshold.

Edit the HARD-CODED SETTINGS section as needed (no CLI args).
"""

import os
import math
from collections import Counter
from typing import Iterator, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# HARD-CODED SETTINGS
# =========================
FASTA_PATH = "../dna.fasta"  # path to your FASTA file
WINDOW     = 9              # sliding window size (bp)
STEP       = 1              # slide step (bp)
OUTDIR     = "results"      # output directory
PREFIX     = None           # optional file prefix; None uses FASTA header
NA_MOLAR   = 0.001          # [Na+] (M) for complex Tm

# Thresholds (°C) for each signal:
THRESHOLDS: Dict[str, float] = {
    "Tm_simple":  25.0,
    "Tm_complex": -10.0,
}
# =========================


# ---------- FASTA ----------
def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
    header = None
    seq_chunks: List[str] = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks).upper()
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, ''.join(seq_chunks).upper()


# ---------- Tm formulas ----------
def computeMeltingTempSimple(seq: str) -> float:
    """Wallace rule: Tm = 4*(G+C) + 2*(A+T); counts only A/C/G/T."""
    s = ''.join(ch for ch in seq.upper() if ch in 'ACGT')
    if not s:
        return float('nan')
    A = s.count("A"); C = s.count("C"); G = s.count("G"); T = s.count("T")
    return 4 * (C + G) + 2 * (A + T)


def computeMeltingTempComplex(seq: str, na_molar: float = NA_MOLAR) -> float:
    """
    Salt-adjusted formula:
    Tm = 81.5 + 16.6*log10([Na+]) + 0.41*(%GC) - 600/length
    """
    s = ''.join(ch for ch in seq.upper() if ch in 'ACGT')
    if not s:
        return float('nan')
    if na_molar <= 0:
        raise ValueError("[Na+] must be positive (mol/L).")
    length = len(s)
    gc_pct = 100.0 * (s.count("G") + s.count("C")) / length
    return 81.5 + 16.6 * math.log10(na_molar) + 0.41 * gc_pct - (600.0 / length)


# ---------- Sliding-window metrics ----------
def window_metrics(seq: str, window: int, step: int = 1, na_molar: float = NA_MOLAR) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      win (1..N), position (center, 1-based), A, C, G, T, Tm_simple, Tm_complex
    """
    n = len(seq)
    data = {
        'win': [],
        'position': [],
        'A': [], 'C': [], 'G': [], 'T': [],
        'Tm_simple': [], 'Tm_complex': []
    }
    valid = set('ACGT')

    win_id = 0
    for start in range(0, n - window + 1, step):
        win_id += 1
        wseq = seq[start:start + window]
        filtered = ''.join(b for b in wseq if b in valid)
        counts = Counter(filtered)
        denom = sum(counts.values())
        if denom == 0:
            a = c = g = t = math.nan
            tm_s = tm_c = math.nan
        else:
            a = 100.0 * counts.get('A', 0) / denom
            c = 100.0 * counts.get('C', 0) / denom
            g = 100.0 * counts.get('G', 0) / denom
            t = 100.0 * counts.get('T', 0) / denom
            tm_s = computeMeltingTempSimple(filtered)
            tm_c = computeMeltingTempComplex(filtered, na_molar)
        center = start + (window // 2) + 1  # 1-based center position

        data['win'].append(win_id)
        data['position'].append(center)
        data['A'].append(a); data['C'].append(c); data['G'].append(g); data['T'].append(t)
        data['Tm_simple'].append(tm_s); data['Tm_complex'].append(tm_c)

    return pd.DataFrame(data)


# ---------- Threshold + plotting helpers ----------
def area_above_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Trapezoidal area where y > thr."""
    above = np.clip(y - thr, 0, None)
    return float(np.trapz(above, x))


def plot_tm_signals_with_thresholds(df: pd.DataFrame,
                                    thresholds: Dict[str, float],
                                    title: str,
                                    out_png: str) -> None:
    """
    Chart 1:
      - Plot both signals as SOLID LINES:
          * Tm_simple  -> BLUE
          * Tm_complex -> ORANGE
      - Add HORIZONTAL THRESHOLD LINES (same colors, dashed).
    """
    x = df['position'].to_numpy(dtype=float)
    y_simple  = df['Tm_simple'].to_numpy(dtype=float)
    y_complex = df['Tm_complex'].to_numpy(dtype=float)

    thr_simple  = thresholds['Tm_simple']
    thr_complex = thresholds['Tm_complex']

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y_simple,  label="Tm_simple",  linewidth=2.0, color='tab:blue')
    plt.plot(x, y_complex, label="Tm_complex", linewidth=2.0, color='tab:orange')

    plt.axhline(thr_simple,  linestyle='--', linewidth=1.5, color='tab:blue',   label=f"threshold simple = {thr_simple:g} °C")
    plt.axhline(thr_complex, linestyle='--', linewidth=1.5, color='tab:orange', label=f"threshold complex = {thr_complex:g} °C")

    plt.xlabel("Window center position (bp)")
    plt.ylabel("Melting temperature (°C)")
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a boolean mask (length N) into contiguous runs of True.
    Returns list of (start_win, end_win) inclusive, 1-based indices.
    """
    runs = []
    start = None
    N = len(mask)
    for i in range(N):
        if mask[i] and start is None:
            start = i + 1
        elif (not mask[i]) and start is not None:
            runs.append((start, i))  # i is last True (exclusive -> i)
            start = None
    if start is not None:
        runs.append((start, N))
    return runs


def plot_regions_above_threshold(df: pd.DataFrame,
                                 thresholds: Dict[str, float],
                                 out_png: str) -> None:
    """
    Chart 2:
      - Two horizontal strips (subplots):
          * Top = P1 (Tm_simple, blue)
          * Bottom = P2 (Tm_complex, orange)
      - Draw filled rectangles for contiguous windows where Tm > threshold.
      - X-axis = window number (1..N).
    """
    wins = df['win'].to_numpy()
    y_simple  = df['Tm_simple'].to_numpy(dtype=float)
    y_complex = df['Tm_complex'].to_numpy(dtype=float)

    thr_simple  = thresholds['Tm_simple']
    thr_complex = thresholds['Tm_complex']

    mask_s = y_simple  > thr_simple
    mask_c = y_complex > thr_complex

    runs_s = _true_runs(mask_s)
    runs_c = _true_runs(mask_c)

    # Convert runs (start,end) to broken_barh ranges: (xmin, width)
    ranges_s = [(start - 0.5, (end - start + 1)) for start, end in runs_s]
    ranges_c = [(start - 0.5, (end - start + 1)) for start, end in runs_c]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4), dpi=120)

    # Top: P1 simple (blue)
    ax1.broken_barh(ranges_s, (0.25, 0.5), facecolors='tab:blue')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.5])
    ax1.set_yticklabels(['P1'])
    ax1.set_title(f"P1 - Regions Above Threshold ({thr_simple:.1f}°C)")
    ax1.grid(False)

    # Bottom: P2 complex (orange)
    ax2.broken_barh(ranges_c, (0.25, 0.5), facecolors='tab:orange')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(['P2'])
    ax2.set_title(f"P2 - Regions Above Threshold ({thr_complex:.1f}°C)")
    ax2.set_xlabel("Window Number")
    ax2.grid(False)

    # X limits (nice padding)
    nwin = int(wins.max()) if len(wins) else 0
    ax2.set_xlim(0, max(1, nwin) + 1)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def sanitize_name(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
    return safe[:80] or 'sequence'


# ---------- Main ----------
def main():
    # Validate settings
    if WINDOW <= 0:
        raise SystemExit("WINDOW must be positive")
    if STEP <= 0:
        raise SystemExit("STEP must be positive")
    if NA_MOLAR <= 0:
        raise SystemExit("NA_MOLAR must be positive (mol/L)")
    for key in ("Tm_simple", "Tm_complex"):
        if key not in THRESHOLDS:
            raise SystemExit(f"THRESHOLDS must define '{key}'")
    if not os.path.exists(FASTA_PATH):
        raise SystemExit(f"FASTA not found: {FASTA_PATH}")

    os.makedirs(OUTDIR, exist_ok=True)
    summary_rows = []

    for idx, (hdr, seq) in enumerate(read_fasta(FASTA_PATH), start=1):
        if len(seq) < WINDOW:
            print(f"[skip] Sequence {idx} ('{hdr}') shorter than window ({len(seq)} < {WINDOW})")
            continue

        df = window_metrics(seq, WINDOW, STEP, NA_MOLAR)
        base_prefix = PREFIX or sanitize_name(hdr or f"seq{idx}")

        # Per-sequence CSV of metrics
        csv_path = os.path.join(OUTDIR, f"{base_prefix}.csv")
        df.to_csv(csv_path, index=False)

        # Chart 1: signals + threshold lines (blue/orange)
        chart1_png = os.path.join(OUTDIR, f"{base_prefix}_signals_with_thresholds.png")
        plot_tm_signals_with_thresholds(
            df, THRESHOLDS,
            title=f"Tm signals with thresholds (W={WINDOW}, step={STEP}, [Na+]={NA_MOLAR} M)\n{hdr}",
            out_png=chart1_png
        )

        # Chart 2: rectangles of windows above thresholds
        chart2_png = os.path.join(OUTDIR, f"{base_prefix}_regions_above_threshold.png")
        plot_regions_above_threshold(df, THRESHOLDS, chart2_png)

        print(f"[ok] {hdr} -> {csv_path}, {chart1_png}, {chart2_png}")

        # Optional: areas above threshold (using center positions)
        x = df['position'].to_numpy(dtype=float)
        a_simple  = area_above_threshold(x, df['Tm_simple'].to_numpy(dtype=float), THRESHOLDS["Tm_simple"])
        a_complex = area_above_threshold(x, df['Tm_complex'].to_numpy(dtype=float), THRESHOLDS["Tm_complex"])
        summary_rows.append({
            "sequence": hdr,
            "window": WINDOW,
            "step": STEP,
            "na_molar": NA_MOLAR,
            "threshold_simple": THRESHOLDS["Tm_simple"],
            "threshold_complex": THRESHOLDS["Tm_complex"],
            "area_above_simple": a_simple,
            "area_above_complex": a_complex,
        })

    # Overall summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTDIR, "threshold_area_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"[ok] Wrote {summary_csv}")


if __name__ == "__main__":
    main()
