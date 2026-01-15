import matplotlib.pyplot as plt
from Bio import Entrez, SeqIO
from typing import List, Tuple

# ------------------------
# REQUIRED for NCBI access
# ------------------------
Entrez.email = "your_email@example.com"

print("STARTING PROGRAM")   # <--- sanity check


# ======================================================
# Download influenza genomes (10 known RefSeq IDs)
# ======================================================

def download_influenza_genomes(n: int) -> List[Tuple[str, str]]:
    known_ids = [
        "NC_002016.1",
        "NC_002017.1",
        "NC_002018.1",
        "NC_002019.1",
        "NC_002020.1",
        "NC_002021.1",
        "NC_002022.1",
        "NC_002023.1",
        "NC_007366.1",
        "NC_007367.1",
    ]

    ids = known_ids[:n]

    print("Fetching genomes from NCBI...")

    handle = Entrez.efetch(
        db="nuccore",
        id=",".join(ids),
        rettype="fasta",
        retmode="text"
    )
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    if len(records) == 0:
        print("ERROR: No genomes downloaded — check internet or firewall")
        return []

    genomes = []
    for r in records:
        genomes.append((r.id, str(r.seq).upper()))

    print(f"Downloaded {len(genomes)} genomes.")
    return genomes


# ======================================================
# Digital stain helpers
# ======================================================

def sliding_windows(seq: str, w: int):
    return [seq[i:i+w] for i in range(len(seq)-w+1)]

def cg_percent(seq: str):
    cg = sum(1 for b in seq if b in "CG")
    return 100 * cg / len(seq)

def kappa_ic(window: str):
    A = window
    N = len(A) - 1
    T = 0
    for shift in range(1, N+1):
        B = A[shift:]
        matches = sum(1 for i in range(len(B)) if A[i] == B[i])
        T += (matches / len(B)) * 100
    return T / N

def digital_stain(seq: str, w: int):
    xs = [cg_percent(win) for win in sliding_windows(seq, w)]
    ys = [kappa_ic(win) for win in sliding_windows(seq, w)]
    return xs, ys

def center(xs, ys):
    return (sum(xs)/len(xs), sum(ys)/len(ys))


# ======================================================
# MAIN
# ======================================================

def main():
    genomes = download_influenza_genomes(10)

    if not genomes:
        print("No genomes to process. Exiting.")
        return

    print("Genomes downloaded:")
    for name, _ in genomes:
        print(" -", name)

    WINDOW = 30

    # -------- Chart 1 --------
    plt.figure(1)
    for name, seq in genomes:
        xs, ys = digital_stain(seq, WINDOW)
        cx, cy = center(xs, ys)
        plt.scatter(xs, ys, s=5, alpha=0.4)

    plt.title("Digital stains of 10 influenza genomes")
    plt.xlabel("(C+G)%")
    plt.ylabel("Kappa IC")
    plt.grid(True)

    # -------- Chart 2 --------
    centers_x = []
    centers_y = []
    labels = []

    for name, seq in genomes:
        xs, ys = digital_stain(seq, WINDOW)
        cx, cy = center(xs, ys)
        centers_x.append(cx)
        centers_y.append(cy)
        labels.append(name)

    plt.figure(2)
    plt.scatter(centers_x, centers_y)

    for x, y, label in zip(centers_x, centers_y, labels):
        plt.annotate(label, (x, y))

    plt.title("Centers of digital stains")
    plt.xlabel("Center (C+G)%")
    plt.ylabel("Center (Kappa IC)")
    plt.grid(True)

    plt.show()


# Must be at bottom!
if __name__ == "__main__":
    main()
