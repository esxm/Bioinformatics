import matplotlib.pyplot as plt
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq

# --- I/O ---
def read_fasta(file_path: str) -> str:
    """Read a single FASTA record and return its raw sequence."""
    with open(file_path, "r", encoding="utf-8") as fh:
        rec = SeqIO.read(fh, "fasta")  # one record expected
    return str(rec.seq)

# --- Normalization ---
def dna_to_rna(seq: str) -> str:
    """Uppercase, convert T->U, drop non-ACGU symbols."""
    s = seq.upper().replace("T", "U")
    return "".join(b for b in s if b in "ACGU")

# --- Counting (frame 0 across entire sequence) ---
def calculate_codon_frequencies(rna_seq: str) -> Counter:
    """Count codons in frame 0 across the whole sequence."""
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq) - 2, 3)]
    return Counter(codons)

# --- Plotting ---
def plot_top_codons(codon_counts: Counter, title: str, top_n: int = 10):
    top = codon_counts.most_common(top_n)
    if not top:
        print(f"(no codons) {title}")
        return
    codons, counts = zip(*top)
    plt.figure(figsize=(10, 6))
    plt.bar(codons, counts)
    plt.title(title)
    plt.xlabel("Codon")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# --- Amino acids (exclude stops via '*') ---
def translate_to_amino_acids(codon_counts: Counter) -> Counter:
    aa_counts = Counter()
    for codon, n in codon_counts.items():
        aa = str(Seq(codon).translate())  # Standard table
        if aa != "*":                     # exclude stop codons
            aa_counts[aa] += n
    return aa_counts

# --- NEW: Ranked "top common codons" between the two genomes ---
def top_common_codons(c1: Counter, c2: Counter, k: int = 10, method: str = "geomean"):
    """
    Rank codons that appear in BOTH genomes by a normalized score.

    method:
      - "geomean": geometric mean of normalized frequencies (default)
      - "min":     min of normalized frequencies
      - "sum":     sum of normalized frequencies

    Returns: list of (codon, covid_count, flu_count, covid_freq, flu_freq, score)
    """
    n1, n2 = sum(c1.values()), sum(c2.values())
    common = set(c1) & set(c2)
    rows = []
    for codon in common:
        f1 = c1[codon] / n1 if n1 else 0.0
        f2 = c2[codon] / n2 if n2 else 0.0
        if method == "geomean":
            score = (f1 * f2) ** 0.5
        elif method == "min":
            score = min(f1, f2)
        elif method == "sum":
            score = f1 + f2
        else:
            raise ValueError("method must be one of: geomean, min, sum")
        rows.append((codon, c1[codon], c2[codon], f1, f2, score))
    rows.sort(key=lambda x: x[-1], reverse=True)
    return rows[:k]

# --- OPTIONAL: Overlap of the two top-10 lists (by rank) ---
def overlap_of_top_lists(c1: Counter, c2: Counter, k: int = 10):
    t1 = [c for c,_ in c1.most_common(k)]
    t2 = [c for c,_ in c2.most_common(k)]
    inter = set(t1) & set(t2)
    r1 = {c:i for i,c in enumerate(t1)}
    r2 = {c:i for i,c in enumerate(t2)}
    # order by combined rank
    return sorted(inter, key=lambda c: r1[c] + r2[c])

# --- Paths (use raw strings if you keep backslashes on Windows) ---
covid_path = r"../covid.fasta"
flu_path   = r"../influenza.fasta"

# --- Run ---
covid_seq = read_fasta(covid_path)
flu_seq   = read_fasta(flu_path)

covid_rna = dna_to_rna(covid_seq)
flu_rna   = dna_to_rna(flu_seq)

covid_codon_counts = calculate_codon_frequencies(covid_rna)
flu_codon_counts   = calculate_codon_frequencies(flu_rna)

# a) & b) charts
plot_top_codons(covid_codon_counts, "Top 10 Codons in SARS-CoV-2 Genome")
plot_top_codons(flu_codon_counts,   "Top 10 Codons in Influenza A Genome")

# c) Top common codons (ranked with normalized score)
top_common = top_common_codons(covid_codon_counts, flu_codon_counts, k=10, method="geomean")
print("Top common codons (normalized, method=geomean):")
for codon, cnt1, cnt2, f1, f2, score in top_common:
    print(f"  {codon}: COVID {cnt1} ({f1:.3%}), Flu {cnt2} ({f2:.3%}), score={score:.6f}")

# (Optional) overlap of the two top-10 lists by rank (not normalized)
overlap = overlap_of_top_lists(covid_codon_counts, flu_codon_counts, k=10)
print("Codons in BOTH top-10 lists (by combined rank):", ", ".join(overlap) if overlap else "(none)")

# d) top 3 amino acids per genome
covid_aa_counts = translate_to_amino_acids(covid_codon_counts)
flu_aa_counts   = translate_to_amino_acids(flu_codon_counts)
print("Top 3 Amino Acids in SARS-CoV-2:", covid_aa_counts.most_common(3))
print("Top 3 Amino Acids in Influenza A:", flu_aa_counts.most_common(3))
