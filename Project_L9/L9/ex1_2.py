#!/usr/bin/env python3
from Bio import Entrez, SeqIO
import math
import matplotlib.pyplot as plt

# ========= HARD-CODED PARAMETERS ========= #

# NCBI requires an email; change this to your real address.
Entrez.email = "your.email@example.com"

# Any nucleotide record that's comfortably > 3000 bp.
NCBI_ACCESSION = "NC_001416"  # bacteriophage lambda (~48 kb)

# Desired subsequence length (between 1000 and 3000 nt)
SUBSEQ_LENGTH = 2000

# Restriction enzymes:
#   name: (recognition_sequence_5to3, cut_offset_from_5prime)
# cut_offset is where the phosphodiester bond is broken in the top strand,
# counted from the beginning of the recognition sequence.
ENZYMES = {
    "EcoRI":   ("GAATTC", 1),  # 5' G^AATTC 3'
    "BamHI":   ("GGATCC", 1),  # 5' G^GATCC 3'
    "HindIII": ("AAGCTT", 1),  # 5' A^AGCTT 3'
    "TaqI":    ("TCGA",   1),  # 5' T^CGA 3'
    "HaeIII":  ("GGCC",   2),  # 5' GG^CC 3'
}

# Lanes on the gel: one single-enzyme digest per lane,
# matching the photo that shows 3 different digests.
DIGESTS = {
    "EcoRI":   ["EcoRI"],
    "BamHI":   ["BamHI"],
    "HindIII": ["HindIII"],
    "TaqI":    ["TaqI"],
    "HaeIII":  ["HaeIII"],
}


# ======================================== #


def fetch_ncbi_sequence(accession: str, subseq_len: int) -> str:
    """
    Fetch a nucleotide record from NCBI and return a subsequence
    of length 'subseq_len' (between 1000 and 3000 nt).
    """
    print(f"[INFO] Fetching {accession} from NCBI...")
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="fasta",
        retmode="text"
    )
    record = SeqIO.read(handle, "fasta")
    handle.close()

    full_seq = str(record.seq).upper().replace("U", "T")
    print(f"[INFO] Full sequence length: {len(full_seq)} bp")

    if subseq_len < 1000 or subseq_len > 3000:
        raise ValueError("subseq_len must be between 1000 and 3000 nt.")

    if len(full_seq) < subseq_len:
        raise ValueError("NCBI sequence is shorter than requested subsequence.")

    # Take a subsequence from the middle to be “arbitrary”
    start = (len(full_seq) - subseq_len) // 2
    end = start + subseq_len
    subseq = full_seq[start:end]
    print(f"[INFO] Using subsequence {start}-{end} (length {len(subseq)} bp)")
    return subseq


def find_cut_sites(seq: str, recognition: str, offset: int):
    """
    Find all cut positions for one enzyme on a single-stranded sequence.
    Returns a list of integer cut indices (0-based, in the original sequence).
    """
    seq = seq.upper()
    recognition = recognition.upper()
    cut_sites = []

    i = 0
    L = len(recognition)
    while i <= len(seq) - L:
        if seq[i:i + L] == recognition:
            cut_sites.append(i + offset)
        i += 1  # allow overlapping sites
    return cut_sites


def digest_sequence(seq: str, enzymes_to_use):
    """
    Perform a digest with one or more enzymes.
    enzymes_to_use: list of enzyme names (keys from ENZYMES).
    Returns sorted fragment sizes (list of ints, largest to smallest).
    """
    cut_positions = {0, len(seq)}  # start and end of the sequence

    for name in enzymes_to_use:
        recog, offset = ENZYMES[name]
        sites = find_cut_sites(seq, recog, offset)
        cut_positions.update(sites)

    cuts_sorted = sorted(cut_positions)
    fragments = []
    for i in range(len(cuts_sorted) - 1):
        frag_len = cuts_sorted[i + 1] - cuts_sorted[i]
        fragments.append(frag_len)

    return sorted(fragments, reverse=True)


def print_digest_report(seq, digests):
    """
    Print a text summary of all digests.
    """
    print("\n=== In-silico restriction digest report ===")
    print(f"Sequence length: {len(seq)} bp\n")

    for lane_name, enzyme_list in digests.items():
        frags = digest_sequence(seq, enzyme_list)
        print(f"Lane: {lane_name}")
        print(f"  Enzyme(s): {', '.join(enzyme_list)}")
        print(f"  Number of fragments: {len(frags)}")
        print(f"  Fragment sizes (bp, largest → smallest):")
        print("   ", frags)
        print()


def simulate_violet_gel(seq, digests):
    """
    Simulate an electrophoresis gel with violet-tinted bands.

    Smaller fragments migrate further (towards +).
    This is a qualitative visualization, not a calibrated gel.
    """
    # Simple migration model: distance ~ a - b * log10(size)
    a = 10.0
    b = 2.0

    fig, ax = plt.subplots(figsize=(3.5, 6))

    # Dark “gel” background & overall style
    fig.patch.set_facecolor("#050008")       # outside the gel
    ax.set_facecolor("#050008")              # inside the gel
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis='y', colors="white")
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    num_lanes = len(digests)
    lane_width = 0.9
    lane_spacing = 1.4

    # Colors: violet bands, slightly brighter wells text, etc.
    band_color = "#c68cff"  # light violet
    label_color = "#f0e6ff"  # pale whitish violet

    for lane_idx, (lane_name, enzymes) in enumerate(digests.items(), start=1):
        frags = digest_sequence(seq, enzymes)
        x_center = lane_idx * lane_spacing

        # Draw “well” at the top as a small rectangle outline
        ax.add_patch(
            plt.Rectangle(
                (x_center - lane_width / 2, a - 0.4),
                lane_width,
                0.3,
                fill=False,
                edgecolor=label_color,
                linewidth=1,
            )
        )

        # Draw each DNA fragment as a horizontal violet band
        for size in frags:
            if size <= 0:
                continue
            distance = a - b * math.log10(size)
            ax.hlines(
                y=distance,
                xmin=x_center - lane_width / 2,
                xmax=x_center + lane_width / 2,
                colors=band_color,
                linewidth=3,
            )

        # Lane label at the top
        ax.text(
            x_center,
            a + 0.7,
            lane_name,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color=label_color,
        )

    ax.invert_yaxis()  # top = wells (negative electrode), bottom = positive
    ax.set_ylabel("Migration (arbitrary units)", color="white")
    ax.set_title("Simulated restriction digest gel", color="white")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Get DNA sequence from NCBI
    seq = fetch_ncbi_sequence(NCBI_ACCESSION, SUBSEQ_LENGTH)

    # 2. Print enzyme info (hard-coded)
    print("Using enzymes (top-strand recognition & cut position):")
    for name, (recog, offset) in ENZYMES.items():
        print(f"  {name:7s}  5'-{recog}-3', cut after index {offset}")
    print()

    # 3. Run digests and print textual report
    print_digest_report(seq, DIGESTS)

    # 4. Show a violet-tinted simulated gel
    simulate_violet_gel(seq, DIGESTS)


if __name__ == "__main__":
    main()
