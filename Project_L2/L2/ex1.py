from itertools import product

# Given sequence
S = "ATTGTCCCAATCTGTTG"

# Nucleotides
nucleotides = ['A', 'T', 'G', 'C']

# Generate all di-nucleotides (2-mers)
dinucleotides = [''.join(p) for p in product(nucleotides, repeat=2)]

# Generate all tri-nucleotides (3-mers)
trinucleotides = [''.join(p) for p in product(nucleotides, repeat=3)]

def calculate_frequency(sequence, kmers):
    """Calculate relative frequencies of k-mers in the sequence"""
    frequencies = {}
    total_count = 0

    for kmer in kmers:
        count = 0
        kmer_len = len(kmer)

        # Search for kmer in sequence (including overlapping occurrences)
        for i in range(len(sequence) - kmer_len + 1):
            if sequence[i:i+kmer_len] == kmer:
                count += 1

        frequencies[kmer] = count
        total_count += count

    # Calculate relative frequencies
    relative_frequencies = {}
    if total_count > 0:
        for kmer, count in frequencies.items():
            relative_frequencies[kmer] = count / total_count
    else:
        relative_frequencies = {kmer: 0.0 for kmer in kmers}

    return frequencies, relative_frequencies

# Calculate frequencies for di-nucleotides
print("\nDI-NUCLEOTIDES")

di_counts, di_relative = calculate_frequency(S, dinucleotides)

print(f"\nSequence: {S}")
print(f"Length: {len(S)}")
print(f"\nTotal possible di-nucleotide positions: {len(S) - 1}")
print(f"\nDi-nucleotide frequencies (count > 0):")
for kmer in sorted(dinucleotides):
    if di_counts[kmer] > 0:
        print(f"{kmer}: count={di_counts[kmer]}, relative_freq={100*di_relative[kmer]:.4f}%")

# Calculate frequencies for tri-nucleotides

print("TRI-NUCLEOTIDES")

tri_counts, tri_relative = calculate_frequency(S, trinucleotides)

print(f"\nTotal possible tri-nucleotide positions: {len(S) - 2}")
print(f"\nTri-nucleotide frequencies (count > 0):")
for kmer in sorted(trinucleotides):
    if tri_counts[kmer] > 0:
        print(f"{kmer}: count={tri_counts[kmer]}, relative_freq={100*tri_relative[kmer]:.4f}%")

# Summary statistics

print("SUMMARY")

print(f"Total di-nucleotides found: {sum(di_counts.values())}")
print(f"Unique di-nucleotides found: {sum(1 for c in di_counts.values() if c > 0)}/{len(dinucleotides)}")
print(f"Total tri-nucleotides found: {sum(tri_counts.values())}")
print(f"Unique tri-nucleotides found: {sum(1 for c in tri_counts.values() if c > 0)}/{len(trinucleotides)}")
