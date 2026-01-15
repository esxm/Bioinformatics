from __future__ import annotations
import sys
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from Bio import Entrez, SeqIO
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# -----------------------------
# USER CONFIG (edit these)
# -----------------------------
EMAIL = "you@example.com"     # REQUIRED by NCBI policy: set to your email
API_KEY = None                # Optional: your NCBI API key
KMIN = 2
KMAX = 6
MIN_LEN = 1000
MAX_LEN = 3000
INFLUENZA_N = 10
TOP_N = 30                    # top-N repeat units (by total frequency) to display
SEED = 42
OUTDIR = "plots"
MIN_CONSECUTIVE_REPEATS = 2   # tandem repeats must appear at least this many times in a row

# -----------------------------
# NCBI / Entrez helpers
# -----------------------------
def set_entrez_credentials(email: str, api_key: str | None = None):
    if not email or "@" not in email:
        raise ValueError("Please provide a valid email for NCBI Entrez")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

def esearch_ids(term: str, db: str = "nuccore", retmax: int = 50) -> List[str]:
    with Entrez.esearch(db=db, term=term, retmax=retmax) as handle:
        rec = Entrez.read(handle)
    return rec.get("IdList", [])

def efetch_fasta(seq_id: str, db: str = "nuccore") -> Tuple[str, str]:
    with Entrez.efetch(db=db, id=seq_id, rettype="fasta", retmode="text") as h:
        fasta_text = h.read()
    from io import StringIO
    handle = StringIO(fasta_text)
    record = next(SeqIO.parse(handle, "fasta"))
    title = record.description
    seq = str(record.seq).upper().replace("U", "T")
    return title, seq

# -----------------------------
# Search terms
# -----------------------------
INFLUENZA_TERM_TEMPLATE = (
    '('
    '"Influenza A virus"[Organism]'
    ' AND (ha[Gene] OR hemagglutinin[Title])'
    ' AND ("complete cds"[Title] OR cds[Title])'
    ' AND 1000:3000[SLEN]'
    ')'
)

RANDOM_GENOMIC_TERM_TEMPLATE = (
    '('
    'biomol_genomic[PROP]'
    ' AND srcdb_refseq[PROP]'
    ' AND 1000:3000[SLEN]'
    ' NOT mitochondrion[Title]'
    ')'
)

# -----------------------------
# Tandem repeat detection
# -----------------------------
def detect_tandem_repeats(seq: str, kmin: int = 6, kmax: int = 10, min_repeats: int = 2) -> Counter:
    """
    Return Counter(unit) = number of *blocks* where that unit repeats consecutively
    (each block counted once, regardless of its length).
    """
    seq = seq.upper().replace("U", "T")
    n = len(seq)
    counts = Counter()
    i = 0
    while i < n:
        advanced = False
        for k in range(kmax, kmin - 1, -1):
            if i + k * min_repeats > n:
                continue
            unit = seq[i:i+k]
            if set(unit) - {"A", "C", "G", "T", "N"}:
                continue
            m = 1
            while i + (m+1)*k <= n and seq[i + m*k : i + (m+1)*k] == unit:
                m += 1
            if m >= min_repeats:
                counts[unit] += 1
                i += m * k  # skip the whole block
                advanced = True
                break
        if not advanced:
            i += 1
    return counts

# -----------------------------
# Fetch helpers
# -----------------------------
def fetch_arbitrary_sequence(min_len: int, max_len: int) -> Tuple[str, str]:
    term = RANDOM_GENOMIC_TERM_TEMPLATE.replace("1000:3000", f"{min_len}:{max_len}")
    ids = esearch_ids(term, retmax=200)
    if not ids:
        raise RuntimeError("No sequences found for the requested length range.")
    seq_id = random.choice(ids)
    return efetch_fasta(seq_id)

def fetch_influenza_sequences(n: int, min_len: int, max_len: int) -> List[Tuple[str, str]]:
    term = INFLUENZA_TERM_TEMPLATE.replace("1000:3000", f"{min_len}:{max_len}")
    ids = esearch_ids(term, retmax=max(50, 5*n))
    if not ids:
        raise RuntimeError("No influenza sequences matched the query.")

    random.shuffle(ids)
    seqs: List[Tuple[str, str]] = []
    seen_titles = set()
    iterator = ids if not TQDM else tqdm(ids, desc="Fetching influenza", unit="seq")
    for seq_id in iterator:
        try:
            title, seq = efetch_fasta(seq_id)
        except Exception:
            continue
        key = tuple(title.split()[0:8])   # light dedup
        if key in seen_titles:
            continue
        seen_titles.add(key)
        seqs.append((title, seq))
        if len(seqs) >= n:
            break
    return seqs

# -----------------------------
# Plotting
# -----------------------------
def plot_multi_counts(series: List[Tuple[str, Counter]], title: str, outfile: Path, top: int = 30):
    """
    Grouped bar chart of top-N tandem repeat units across multiple genomes.
    `series` is a list of (label, Counter).
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Top-N units by *total* frequency across all genomes
    total = Counter()
    for _, cnt in series:
        total.update(cnt)

    if not total:
        plt.figure(figsize=(10, 4))
        plt.title(f"{title} — no tandem repeats found")
        plt.xlabel("repeat unit")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)
        plt.close()
        return

    labels = [k for k, _ in total.most_common(top)]
    x = np.arange(len(labels))
    n_series = max(1, len(series))
    width = min(0.8 / n_series, 0.25)  # keep bars readable

    plt.figure(figsize=(max(12, min(24, 0.6 * len(labels) + 4)), 6))
    for i, (name, cnt) in enumerate(series):
        vals = [cnt.get(k, 0) for k in labels]
        plt.bar(x + i * width, vals, width=width, label=name)

    # Center tick labels
    plt.xticks(x + (n_series - 1) * width / 2, labels, rotation=90)
    plt.title(title)
    plt.xlabel("repeat unit (tandem; k=6..10)")
    plt.ylabel("frequency (blocks)")
    plt.legend(title="Genome")
    plt.tight_layout()
    plt.savefig(outfile, dpi=220)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    random.seed(SEED)
    set_entrez_credentials(EMAIL, API_KEY)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Arbitrary DNA sequence (1,000–3,000 nt)
    print("Fetching arbitrary DNA sequence from NCBI...", file=sys.stderr)
    title, seq = fetch_arbitrary_sequence(MIN_LEN, MAX_LEN)
    print(f"Fetched: {title} (len={len(seq)})", file=sys.stderr)

    # Analyze arbitrary sequence
    print(f"Analyzing tandem repeats (k={KMIN}..{KMAX})...", file=sys.stderr)
    acounts = detect_tandem_repeats(seq, kmin=KMIN, kmax=KMAX, min_repeats=MIN_CONSECUTIVE_REPEATS)

    # 2) Download influenza genomes (HA) and analyze
    print(f"\nFetching {INFLUENZA_N} influenza HA sequences...", file=sys.stderr)
    influenza = fetch_influenza_sequences(INFLUENZA_N, MIN_LEN, MAX_LEN)
    print(f"Got {len(influenza)} sequences.", file=sys.stderr)

    series: List[Tuple[str, Counter]] = [("arbitrary", acounts)]

    for i, (ititle, iseq) in enumerate(influenza, start=1):
        print(f"Analyzing influenza {i}/{len(influenza)}", file=sys.stderr)
        icounts = detect_tandem_repeats(iseq, kmin=KMIN, kmax=KMAX, min_repeats=MIN_CONSECUTIVE_REPEATS)
        # Shorter label for legend
        short = (ititle.split("|")[0] or f"influenza_{i}")[:30]
        series.append((short, icounts))

    # 3) One combined figure
    combined_title = f"Tandem repeats across genomes (k={KMIN}..{KMAX}, min×{MIN_CONSECUTIVE_REPEATS})"
    combined_out = outdir / "combined_tandem_repeats.png"
    plot_multi_counts(series, combined_title, combined_out, top=TOP_N)
    print(f"Combined plot saved: {combined_out}", file=sys.stderr)

    print("\nAll done. Plots are in:", outdir.resolve(), file=sys.stderr)

if __name__ == "__main__":
    main()
