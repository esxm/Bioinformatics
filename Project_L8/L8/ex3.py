from Bio import Entrez, SeqIO
import bisect
import datetime

# ==========================
# 0. SETTINGS
# ==========================
Entrez.email = "your_email@example.com"   # <- PUT YOUR EMAIL HERE


# ==========================
# 1. DOWNLOAD GENOME FROM NCBI
# ==========================

def download_genome_fasta(accession):
    with Entrez.efetch(
        db="nuccore",
        id=accession,
        rettype="fasta",
        retmode="text"
    ) as handle:
        record = SeqIO.read(handle, "fasta")
    return record


# ==========================
# 2. BASIC SEQUENCE UTILITIES
# ==========================

COMPLEMENT = str.maketrans("ACGTacgtnN", "TGCAtgcanN")

def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


# ==========================
# 3. BUILD KMER INDEX
# ==========================

def build_kmer_index(seq, k):
    seq = str(seq).upper()
    index = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        index.setdefault(kmer, []).append(i)
    return index


# ==========================
# 4. FIND INVERTED REPEAT PAIRS
# ==========================

def find_ir_pairs(seq,
                  min_len,
                  max_len,
                  min_spacer=20,
                  max_spacer=500,
                  max_pairs=20):
    """
    Returns a small list of IR pairs (candidate transposons).
    """
    seq = str(seq).upper()
    n = len(seq)
    pairs = []

    for L in range(min_len, max_len + 1):
        index = build_kmer_index(seq, L)

        for i in range(n - L + 1):
            if len(pairs) >= max_pairs:
                return pairs

            left = seq[i:i + L]
            rc = reverse_complement(left)

            if rc not in index:
                continue

            # possible right IR positions
            min_pos = i + L + min_spacer
            max_pos = i + L + max_spacer
            positions = index[rc]

            lo = bisect.bisect_left(positions, min_pos)
            hi = bisect.bisect_right(positions, max_pos)

            for j in positions[lo:hi]:
                spacer = j - (i + L)
                pairs.append({
                    "left_start": i + 1,   # 1-based
                    "right_start": j + 1,  # 1-based
                    "length": L,
                    "spacer": spacer
                })

                if len(pairs) >= max_pairs:
                    return pairs

    return pairs


# ==========================
# 5. FORMAT REPORT FOR ONE GENOME
# ==========================

def format_genome_report(record, accession, min_len, max_len, pairs):
    """
    Build the text block for a single genome (info + hits + interpretation).
    """
    lines = []

    lines.append("--------------------------------------------------")
    lines.append(f"Genome: {record.id}  (Accession: {accession})")
    lines.append("--------------------------------------------------")
    lines.append(f"Genome length: {len(record.seq):,} bp")
    lines.append(f"Inverted repeat length range: {min_len}-{max_len} bp")
    lines.append(f"Max pairs reported: {len(pairs)}")
    lines.append("")

    lines.append("Detected inverted-repeat pairs (candidate transposons):")
    if not pairs:
        lines.append("  None detected with these parameters.")
    else:
        for i, p in enumerate(pairs, 1):
            left_end = p["left_start"] + p["length"] - 1
            right_end = p["right_start"] + p["length"] - 1
            size = right_end - p["left_start"] + 1
            lines.append(
                f"  [#{i}] Left IR: {p['left_start']}-{left_end}, "
                f"Right IR: {p['right_start']}-{right_end}, "
                f"IR length={p['length']}, spacer={p['spacer']}, "
                f"candidate size={size} bp"
            )

    lines.append("")
    lines.append("Interpretation for this genome:")

    if not pairs:
        lines.append(
            "  No inverted-repeat pairs were found within the specified length\n"
            "  and spacer range. This may mean that this genome does not contain\n"
            "  transposons with such short IRs, or that different parameters\n"
            "  (longer IRs or a wider spacer range) are required to detect them."
        )
    else:
        lines.append(
            "  Several inverted-repeat pairs were identified. Each pair consists\n"
            "  of two short sequences that are reverse complements of each other,\n"
            "  separated by an internal fragment of DNA (the candidate transposon).\n"
            "  This structural organization is typical of bacterial insertion\n"
            "  sequences (IS elements) and other transposons.\n"
        )
        lines.append(
            "  Although IR pairs alone are not definitive proof of active\n"
            "  transposons, they strongly suggest transposon-like regions.\n"
            "  To confirm, one would usually:\n"
            "    -search the internal region for transposase ORFs,\n"
            "    -compare the sequence with known IS elements (e.g. ISFinder),\n"
            "    -and look for target-site duplications at insertion boundaries."
        )

    lines.append("")
    return "\n".join(lines)


# ==========================
# 6. MAIN ANALYSIS
# ==========================

def analyze_genome(accession, min_len, max_len):
    """
    Download genome, find IR pairs, and return (record, pairs).
    """
    print(f"\n=== Processing {accession} ===")
    record = download_genome_fasta(accession)
    pairs = find_ir_pairs(
        record.seq,
        min_len=min_len,
        max_len=max_len,
        min_spacer=20,
        max_spacer=500,
        max_pairs=20
    )
    print(f"  Found {len(pairs)} candidate IR pairs.")
    return record, pairs


# ==========================
# 7. MAIN – SINGLE TXT OUTPUT
# ==========================

if __name__ == "__main__":
    print("Choose the inverted repeat (IR) lengths to search for:")
    min_len = int(input("  Minimum IR length (e.g. 4): "))
    max_len = int(input("  Maximum IR length (e.g. 6): "))

    if max_len < min_len:
        raise ValueError("Maximum IR length must be >= minimum IR length")

    # 3 small bacterial genomes
    accessions = [
        "NC_000915.1",   # Helicobacter pylori 26695
        "NC_002163.1",   # Campylobacter jejuni NCTC 11168
        "NC_002162.1"    # Ureaplasma parvum serovar 3
    ]

    all_reports = []
    genome_summaries = []

    for acc in accessions:
        record, pairs = analyze_genome(acc, min_len, max_len)
        all_reports.append(format_genome_report(record, acc, min_len, max_len, pairs))
        genome_summaries.append((record.id, acc, len(record.seq), len(pairs)))

    # Build full combined report
    output_filename = "transposon_results_all_genomes.txt"
    with open(output_filename, "w") as f:
        f.write("=========================================\n")
        f.write("  Bioinformatics Lab – Transposon Search\n")
        f.write("=========================================\n\n")
        
        f.write(f"Generation time: {datetime.datetime.now()}\n")
        f.write(f"IR length range: {min_len}-{max_len} bp\n")
        f.write("Genomes analyzed:\n")
        for gid, acc, glen, npairs in genome_summaries:
            f.write(
                f"  - {gid} (Accession {acc}, {glen:,} bp) "
                f"-> {npairs} candidate IR pairs\n"
            )
        f.write("\n")

        # Per-genome sections
        for rep in all_reports:
            f.write(rep)
            f.write("\n\n")

        # Overall conclusion section
        f.write("=========================================\n")
        f.write("Overall Interpretation and Conclusions\n")
        f.write("=========================================\n\n")
        f.write(
            "In all three bacterial genomes, inverted-repeat (IR) detection was\n"
            "performed without prior knowledge of transposon positions. IRs of\n"
            f"length {min_len}-{max_len} bp were scanned, and potential transposon\n"
            "boundaries were identified wherever pairs of reverse-complement IRs\n"
            "were found separated by a spacer of 20–500 bp.\n\n"
        )

        f.write(
            "The presence of such IR pairs suggests potential insertion sequences\n"
            "(IS elements) or related transposon-like structures. However, IR\n"
            "detection alone is only the first step. To fully validate transposon\n"
            "candidates, further analysis would typically include:\n"
            "  - identification of transposase genes within the candidate region,\n"
            "  - comparison with known transposable elements via BLAST/ISFinder,\n"
            "  - and examination of target-site duplications.\n\n"
        )

        f.write(
            "From a computational perspective, this analysis demonstrates how\n"
            "pattern-based searching (in this case, short inverted repeats) can be\n"
            "used to generate hypotheses about mobile genetic elements in bacterial\n"
            "genomes, without requiring prior annotation of transposons."
        )

    print(f"\nAll done. Combined report saved as: {output_filename}\n")