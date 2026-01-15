#!/usr/bin/env python3
"""
Sliding-window nucleotide composition and melting temperature (Tm) from a FASTA file.

- Scans each sequence using a sliding window of size W (hardcoded below)
- For every window, computes %A, %C, %G, %T and melting temperatures using:
  - Wallace rule (simple): Tm = 4*(G+C) + 2*(A+T)
  - Salt-adjusted (complex): Tm = 81.5 + 16.6*log10([Na+]) + 0.41*%GC - 600/length
- Plots Tm (simple & complex) vs position on the SAME figure (x = window center, 1-based)
- Saves a CSV with all values and a PNG chart per sequence

Notes:
- Non-ACGT characters are ignored for window calculations.
- If a window has zero valid A/C/G/T bases, values are set to NaN.
- For multi-FASTA files, one CSV and one PNG are written per sequence.

EDIT THESE CONSTANTS to use different settings (no command-line args needed).
"""

import os
import math
from collections import Counter
from typing import Iterator, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# HARD-CODED SETTINGS
# =========================
FASTA_PATH = "../dna.fasta"  # <- path to your FASTA
WINDOW     = 9              # sliding window size (bp)
STEP       = 1              # slide step (bp)
OUTDIR     = "results"      # output directory
PREFIX     = None           # optional prefix for outputs; None uses FASTA header
NA_MOLAR   = 0.001          # [Na+] in mol/L for complex Tm
# =========================


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


# --- Helper functions ---
def computeMeltingTempSimple(seq: str) -> float:
    """Wallace rule uses COUNTS (A/T/G/C only)."""
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


# --- Sliding-window calculations ---
def window_metrics(seq: str, window: int, step: int = 1, na_molar: float = NA_MOLAR) -> pd.DataFrame:
    """Compute %A,%C,%G,%T and both Tm formulas for each sliding window.

    Returns a DataFrame with columns:
      position (center, 1-based), A, C, G, T, Tm_simple, Tm_complex
    """
    n = len(seq)
    data = {
        'position': [],
        'A': [], 'C': [], 'G': [], 'T': [],
        'Tm_simple': [], 'Tm_complex': []
    }
    valid = set('ACGT')

    for start in range(0, n - window + 1, step):
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
        data['position'].append(center)
        data['A'].append(a); data['C'].append(c); data['G'].append(g); data['T'].append(t)
        data['Tm_simple'].append(tm_s); data['Tm_complex'].append(tm_c)

    return pd.DataFrame(data)


def plot_tm_both(df: pd.DataFrame, title: str, out_png: str) -> None:
    """Plot BOTH melting temperature curves (simple & complex) vs position on the same figure."""
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(df['position'], df['Tm_simple'], label="Tm (simple / Wallace)", linestyle='--')
    plt.plot(df['position'], df['Tm_complex'], label="Tm (complex / salt-adjusted)")
    plt.xlabel('Sequence position (bp)')
    plt.ylabel('Melting temperature (°C)')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def sanitize_name(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
    return safe[:80] or 'sequence'


def main():
    # Basic validation of hardcoded settings
    if WINDOW <= 0:
        raise SystemExit('WINDOW must be a positive integer')
    if STEP <= 0:
        raise SystemExit('STEP must be a positive integer')
    if NA_MOLAR <= 0:
        raise SystemExit('NA_MOLAR must be positive (mol/L)')
    if not os.path.exists(FASTA_PATH):
        raise SystemExit(f"FASTA not found: {FASTA_PATH}")

    os.makedirs(OUTDIR, exist_ok=True)

    for idx, (hdr, seq) in enumerate(read_fasta(FASTA_PATH), start=1):
        if len(seq) < WINDOW:
            print(f"[skip] Sequence {idx} ('{hdr}') shorter than window ({len(seq)} < {WINDOW})")
            continue
        df = window_metrics(seq, WINDOW, STEP, NA_MOLAR)
        base_prefix = PREFIX or sanitize_name(hdr or f'seq{idx}')
        csv_path = os.path.join(OUTDIR, f"{base_prefix}.csv")
        png_path = os.path.join(OUTDIR, f"{base_prefix}.png")
        df.to_csv(csv_path, index=False)
        title = f"Sliding-window Tm (both) W={WINDOW}, step={STEP}, [Na+]={NA_MOLAR} M\n{hdr}"
        plot_tm_both(df, title, png_path)
        print(f"[ok] Wrote {csv_path} and {png_path}")


if __name__ == '__main__':
    main()
