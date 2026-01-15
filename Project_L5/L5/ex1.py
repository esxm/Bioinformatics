import random
import re
import sys
import urllib.request
import urllib.error
import ssl
import matplotlib.pyplot as plt

# ----------------------------
# Config (tweak these as needed)
# ----------------------------
ACCESSION = "NR_024570.1"      # Any NCBI/ENA nucleotide accession
N_READS = 2000                 # (a) number of random samples
READ_MIN = 100                 # min read length
READ_MAX = 150                 # max read length
MIN_OVERLAP = 10               # minimum overlap to chain reads
PLOT_READS = 120               # how many sampled reads to visualize
RANDOM_SEED = 7                # reproducibility

# ----------------------------
# FASTA fetching (internet)
# ----------------------------
def fetch_fasta_text_from_ena(accession: str) -> str:
    """
    Try EBI/ENA FASTA first (simple, no API key). Returns FASTA text.
    """
    url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{accession}?download=true"
    return http_get(url)

def fetch_fasta_text_from_ncbi(accession: str) -> str:
    """
    Fallback: NCBI E-utilities efetch FASTA. Returns FASTA text.
    """
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nucleotide&id={accession}&rettype=fasta&retmode=text"
    )
    return http_get(url)

def http_get(url: str) -> str:
    """
    Simple HTTP GET with a permissive SSL context for demo convenience.
    """
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTPError fetching {url}: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError fetching {url}: {e.reason}")

def parse_fasta_sequence(fasta_text: str) -> str:
    """
    Parse the first sequence from FASTA text. Returns uppercase A/C/G/T only.
    """
    if not fasta_text or ">" not in fasta_text:
        raise ValueError("Not valid FASTA text (no '>').")
    lines = fasta_text.strip().splitlines()
    seq_lines = []
    for line in lines:
        if line.startswith(">"):
            continue
        seq_lines.append(line.strip())
    seq = "".join(seq_lines).upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "", seq)
    if not seq:
        raise ValueError("Parsed FASTA has no A/C/G/T content after cleaning.")
    return seq

def get_sequence_from_internet(accession: str) -> str:
    """
    Try ENA, then NCBI. Clean to A/C/G/T. If longer than 3000 nt,
    take a random window (1000–3000); if shorter than 1000, raise.
    """
    try:
        fasta = fetch_fasta_text_from_ena(accession)
    except Exception:
        fasta = fetch_fasta_text_from_ncbi(accession)
    seq = parse_fasta_sequence(fasta)

    if len(seq) < 1000:
        raise ValueError(f"Sequence too short ({len(seq)} nt). Choose a longer accession.")
    if len(seq) > 3000:
        random_len = random.randint(1000, 3000)
        start = random.randint(0, len(seq) - random_len)
        seq = seq[start:start + random_len]
    return seq

# ----------------------------
# Read sampling (like your code)
# ----------------------------
def sample_reads(seq: str, n_reads: int, read_min: int, read_max: int, seed: int):
    rng = random.Random(seed)
    samples = []
    positions = []
    for _ in range(n_reads):
        start = rng.randint(0, len(seq) - read_max)
        end = start + rng.randint(read_min, read_max)
        samples.append(seq[start:end])
        positions.append((start, end))
    return samples, positions

# ----------------------------
# Naive greedy reconstruction (like your code)
# ----------------------------
def naive_chain(samples, min_overlap: int):
    """
    Start from samples[0], then for each subsequent sample s, if
    the current 'reconstructed' ends with a prefix of s of length >= min_overlap,
    append the non-overlapping tail of s.
    (Order-dependent, intentionally simple to mirror your example.)
    """
    if not samples:
        return ""
    reconstructed = samples[0]
    for s in samples[1:]:
        max_try = min(len(s), len(reconstructed))
        joined = False
        for overlap in range(max_try, min_overlap - 1, -1):
            if reconstructed.endswith(s[:overlap]):
                reconstructed += s[overlap:]
                joined = True
                break
        if not joined:
            pass
    return reconstructed

# ----------------------------
# Plotting helpers (span map only)
# ----------------------------
def plot_sample_spans(positions, ref_len: int, title: str, n_to_plot: int = 120, seed: int = 11):
    rng = random.Random(seed)
    subset = positions if len(positions) <= n_to_plot else rng.sample(positions, n_to_plot)
    plt.figure(figsize=(12, 5))
    for i, (start, end) in enumerate(subset):
        plt.hlines(i, start, end, linewidth=2)
    plt.title(title)
    plt.xlabel(f"Position along original DNA (0 → {ref_len})")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(RANDOM_SEED)

    # A) Fetch internet FASTA
    try:
        dna = get_sequence_from_internet(ACCESSION)
        print(f"Fetched accession: {ACCESSION}")
    except Exception as e:
        print(f"Failed to fetch {ACCESSION}: {e}")
        sys.exit(1)

    print("Sequence length:", len(dna))

    # B) Sample reads + keep true positions
    samples, positions = sample_reads(
        dna, n_reads=N_READS, read_min=READ_MIN, read_max=READ_MAX, seed=RANDOM_SEED
    )

    # C) Naive reconstruction (order dependent; min overlap ≥ MIN_OVERLAP)
    reconstructed = naive_chain(samples, MIN_OVERLAP)

    # D) Print preview
    print("\n ORIGINAL DNA (first 200 bases):")
    print(dna[:200])
    print("\n RECONSTRUCTED DNA (first 200 bases):")
    print(reconstructed[:200])

    print("\nOriginal length:", len(dna))
    print("Reconstructed length:", len(reconstructed))
    print("Reconstructed is substring of original:", reconstructed in dna)
    print("Original is substring of reconstructed:", dna in reconstructed)

    # E) Plot: span map for a subset of sampled reads (coverage plot removed)
    plot_sample_spans(
        positions,
        ref_len=len(dna),
        title="Internet FASTA: random samples (each line = one fragment)",
        n_to_plot=PLOT_READS,
        seed=RANDOM_SEED
    )

    # F) Discussion
    print("\n MAIN PROBLEM (recap):")
    print(f"""
- This naive chaining depends on read order and only appends when the next read
  overlaps the current end by ≥{MIN_OVERLAP} nt, so many reads never join.
- Repeats/low-complexity regions create ambiguous overlaps.
- Even with high average coverage, local gaps break contiguity.
Solutions: use smarter overlap-layout-consensus or de Bruijn graph assembly,
increase the minimum overlap, and/or reorder reads by best overlap score.
""")

if __name__ == "__main__":
    main()
