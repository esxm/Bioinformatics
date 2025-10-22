#!/usr/bin/env python3
"""
Sliding-window nucleotide composition and melting temperature (Tm) from a FASTA file.

- Scans each sequence using a sliding window of size W (default: 9)
- For every window, computes %A, %C, %G, %T and melting temperatures using:
  - Wallace rule (simple): Tm = 4*(G+C) + 2*(A+T)
  - Empirical (complex):   Tm = 81.5 + 16.6*log10([Na+]) + 0.41*%GC - 600/length
- Plots **Tm vs position** (default: complex formula) with x = window center (1‑based)
- Saves a CSV with all values and a PNG chart per sequence

Usage:
  python sliding_window_composition.py input.fasta \
         --window 9 --step 1 --outdir results --tm complex --na 0.001

Notes:
- Non-ACGT characters are ignored for window calculations.
- If a window has zero valid A/C/G/T bases, values are set to NaN.
- For multi-FASTA files, one CSV and one PNG are written per sequence.
"""

import argparse
import os
import math
from collections import Counter
from typing import Iterator, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd


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


# --- Provided helper functions (integrated) ---
def computeFreq(seq: str, c: str) -> float:
    """Frequency of base c in seq (0..1)."""
    c = c.upper()
    seq = seq.upper()
    if not seq:
        return 0.0
    return seq.count(c) / len(seq)


def computeCount(seq: str, c: str) -> int:
    """Count of base c in seq."""
    return seq.upper().count(c.upper())


def computeMeltingTempSimple(seq: str) -> float:
    # Wallace rule uses COUNTS, not frequencies
    s = ''.join(ch for ch in seq.upper() if ch in 'ACGT')
    A = s.count("A")
    C = s.count("C")
    G = s.count("G")
    T = s.count("T")
    return float('nan') if len(s) == 0 else 4 * (C + G) + 2 * (A + T)


def computeMeltingTempComplex(seq: str, na_molar: float = 0.001) -> float:
    s = ''.join(ch for ch in seq.upper() if ch in 'ACGT')
    if na_molar <= 0:
        raise ValueError("[Na+] must be positive (mol/L).")
    length = len(s)
    if length == 0:
        return float('nan')
    gc_pct = 100.0 * (s.count("G") + s.count("C")) / length
    return 81.5 + 16.6 * math.log10(na_molar) + 0.41 * gc_pct - (600.0 / length)


# --- Sliding-window calculations ---
def window_metrics(seq: str, window: int, step: int = 1, na_molar: float = 0.001) -> pd.DataFrame:
    """Compute %A,%C,%G,%T and Tm for each sliding window.

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


def plot_tm(df: pd.DataFrame, title: str, out_png: str, which: str = 'complex') -> None:
    """Plot melting temperature vs position and save to PNG.

    which: 'complex' (default) or 'simple'
    """
    ycol = 'Tm_complex' if which == 'complex' else 'Tm_simple'
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(df['position'], df[ycol], label=f"Tm ({which})")
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
    ap = argparse.ArgumentParser(description='Sliding-window composition and melting temperature plotter')
    ap.add_argument('fasta', help='Input FASTA file')
    ap.add_argument('--window', type=int, default=9, help='Sliding window size (default: 9)')
    ap.add_argument('--step', type=int, default=1, help='Slide step (default: 1)')
    ap.add_argument('--outdir', default='results', help='Output directory (default: results)')
    ap.add_argument('--prefix', default=None, help='Optional file prefix for outputs')
    ap.add_argument('--tm', choices=['simple','complex'], default='complex', help='Which Tm to plot (default: complex)')
    ap.add_argument('--na', type=float, default=0.001, help='[Na+] in mol/L for complex Tm (default: 0.001)')

    args = ap.parse_args()

    if args.window <= 0:
        raise SystemExit('--window must be a positive integer')
    if args.step <= 0:
        raise SystemExit('--step must be a positive integer')
    if args.na <= 0:
        raise SystemExit('--na must be positive (mol/L)')

    os.makedirs(args.outdir, exist_ok=True)

    for idx, (hdr, seq) in enumerate(read_fasta(args.fasta), start=1):
        if len(seq) < args.window:
            print(f"[skip] Sequence {idx} ('{hdr}') shorter than window ({len(seq)} < {args.window})")
            continue
        df = window_metrics(seq, args.window, args.step, args.na)
        base_prefix = args.prefix or sanitize_name(hdr or f'seq{idx}')
        csv_path = os.path.join(args.outdir, f"{base_prefix}.csv")
        png_path = os.path.join(args.outdir, f"{base_prefix}.png")
        df.to_csv(csv_path, index=False)
        title = f"Sliding-window Tm ({args.tm}) W={args.window}\n{hdr}"
        plot_tm(df, title, png_path, which=args.tm)
        print(f"[ok] Wrote {csv_path} and {png_path}")


if __name__ == '__main__':
    main()

