#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import random
import textwrap
from collections import defaultdict, Counter

import requests
import numpy as np
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
NCBI_EMAIL = "your_email@example.com"   # <-- set this to comply with NCBI policy
NCBI_API_KEY = None                     # optional, if you have one
N_GENOMES = 10
QUERY = 'txid10239[Organism:exp] AND "complete genome"[Title] AND srcdb_refseq[PROP]'
MAX_RETRIES = 4
RETRY_BACKOFF = 1.6

# Synthetic sampling/assembly parameters
READ_LEN = 150           # length of synthetic reads
COVERAGE = 20            # ~X coverage to simulate
K = 31                   # k for de Bruijn graph
RANDOM_SEED = 42
# -----------------------------------

random.seed(RANDOM_SEED)

def ncbi_get(url, params, timeout=30):
    """GET with retries + simple backoff."""
    if NCBI_EMAIL is None or "@" not in NCBI_EMAIL:
        raise ValueError("Please set NCBI_EMAIL to a valid email.")
    base_params = {"tool": "mini-viral-assembler", "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        base_params["api_key"] = NCBI_API_KEY
    params = {**base_params, **params}
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(RETRY_BACKOFF ** i)
    raise last_err

def ncbi_search_viral_genomes(n=N_GENOMES):
    """Search NCBI nuccore for complete viral genomes and return a list of Ids."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    r = ncbi_get(url, {
        "db": "nuccore",
        "term": QUERY,
        "retmode": "json",
        "retmax": n,
        "sort": "relevance",
    })
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise RuntimeError("No genome IDs found from NCBI search.")
    return ids[:n]

def ncbi_fetch_fasta(nuccore_ids):
    """Fetch FASTA records for the given IDs. Returns list of (header, seq)."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    r = ncbi_get(url, {
        "db": "nuccore",
        "id": ",".join(nuccore_ids),
        "rettype": "fasta",
        "retmode": "text"
    })
    text = r.text
    records = []
    header = None
    seq_lines = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header and seq_lines:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip().upper())
    if header and seq_lines:
        records.append((header, "".join(seq_lines)))
    return records

def gc_percent(seq):
    gc = sum(1 for b in seq if b in ("G", "C"))
    at = sum(1 for b in seq if b in ("A", "T"))
    tot = gc + at
    return (100.0 * gc / tot) if tot else 0.0

def make_reads_from_genome(seq, read_len=READ_LEN, coverage=COVERAGE):
    """Create synthetic reads by sampling consecutive windows across the genome."""
    # Estimate number of reads = coverage * genome_len / read_len
    n_reads = max(1, int(round(coverage * len(seq) / read_len)))
    reads = []
    if len(seq) < read_len:
        # edge case: super short genomes
        return [seq] * n_reads
    # Simple sampling: random start positions
    for _ in range(n_reads):
        start = random.randint(0, len(seq) - read_len)
        read = seq[start:start + read_len]
        reads.append(read)
    return reads

# -------- Tiny de Bruijn Assembler (toy) --------
def build_dbg(reads, k):
    edges = defaultdict(Counter)
    indeg = Counter()
    outdeg = Counter()
    for r in reads:
        if len(r) < k: 
            continue
        for i in range(len(r) - k + 1):
            kmer = r[i:i+k]
            if "N" in kmer: 
                continue
            prefix = kmer[:-1]
            suffix = kmer[1:]
            edges[prefix][suffix] += 1
            outdeg[prefix] += 1
            indeg[suffix] += 1
    return edges, indeg, outdeg

def assemble_contigs(edges):
    """Greedy walk to extract contigs from a DBG. Not biologically perfect—just a timing proxy."""
    contigs = []
    visited = set()
    for node in list(edges.keys()):
        if node in visited:
            continue
        # Extend forward
        seq = node
        cur = node
        visited.add(cur)
        while edges.get(cur):
            # pick the most supported edge
            nxt = edges[cur].most_common(1)[0][0]
            seq += nxt[-1]
            cur = nxt
            visited.add(cur)
            # Prevent runaway loops in tiny demo
            if len(seq) > 2_000_000:
                break
        contigs.append(seq)
    return contigs

def run_toy_assembly(reads, k):
    t0 = time.perf_counter()
    edges, indeg, outdeg = build_dbg(reads, k)
    contigs = assemble_contigs(edges)
    ms = (time.perf_counter() - t0) * 1000.0
    # A minimal quality proxy: N50-ish heuristic
    lens = sorted((len(c) for c in contigs), reverse=True)
    n50 = 0
    if lens:
        half = sum(lens) / 2
        running = 0
        for L in lens:
            running += L
            if running >= half:
                n50 = L
                break
    return ms, n50, len(contigs)
# ------------------------------------------------

def main():
    os.makedirs("out/genomes", exist_ok=True)
    os.makedirs("out/reads", exist_ok=True)
    os.makedirs("out/results", exist_ok=True)

    print("Searching NCBI for viral complete genomes...")
    ids = ncbi_search_viral_genomes(N_GENOMES)
    print(f"Found {len(ids)} ids: {ids}")

    print("Downloading FASTA records...")
    records = ncbi_fetch_fasta(ids)
    if len(records) < N_GENOMES:
        print(f"Warning: fetched only {len(records)} genomes.")

    results = []
    for idx, (hdr, seq) in enumerate(records, start=1):
        # Save genome
        fasta_name = f"out/genomes/virus_{idx:02d}.fasta"
        with open(fasta_name, "w") as f:
            f.write(f">{hdr}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + "\n")

        # Compute GC
        gc = gc_percent(seq)

        # Make reads (samples)
        reads = make_reads_from_genome(seq, READ_LEN, COVERAGE)
        reads_path = f"out/reads/virus_{idx:02d}_reads.fq"
        with open(reads_path, "w") as f:
            # FASTQ-ish (no qualities, placeholder)
            for j, r in enumerate(reads, start=1):
                f.write(f"@v{idx}_r{j}\n{r}\n+\n{'I'*len(r)}\n")

        # Assemble + time
        ms, n50, ncontigs = run_toy_assembly(reads, K)

        # Keep stats
        results.append({
            "index": idx,
            "ncbi_header": hdr,
            "genome_len": len(seq),
            "gc_percent": gc,
            "assembly_time_ms": ms,
            "n50_approx": n50,
            "n_contigs": ncontigs,
            "n_reads": len(reads),
        })
        print(f"[{idx:02d}] len={len(seq)} bp | GC={gc:.2f} % | reads={len(reads)} | time={ms:.1f} ms | N50~{n50} | contigs={ncontigs}")

    # Save machine-readable table (JSON) for convenience
    table_path = "out/results/summary.json"
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot: X=GC%, Y=time [ms] — display, don't save
    xs = [r["gc_percent"] for r in results]
    ys = [r["assembly_time_ms"] for r in results]
    labels = [f"V{r['index']:02d}" for r in results]
    lens_bp = [r["genome_len"] for r in results]

    def fmt_bp(n):
        return f"{n:,} bp"

    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.scatter(xs, ys)
    for x, y, label, L in zip(xs, ys, labels, lens_bp):
        # label each point with virus index AND sequence length
        plt.text(x, y, f"{label} • {fmt_bp(L)}", fontsize=8, ha="left", va="bottom")
    plt.xlabel("Overall GC (%)")
    plt.ylabel("Assembly time (ms)")
    plt.title("Toy assembly timing vs GC% (10 viral genomes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Build a plain-text explanation (also print to console)
    gc_arr = np.array(xs)
    time_arr = np.array(ys)
    corr = float(np.corrcoef(gc_arr, time_arr)[0, 1]) if len(xs) > 1 else float("nan")

    fastest = min(results, key=lambda r: r["assembly_time_ms"])
    slowest = max(results, key=lambda r: r["assembly_time_ms"])
    lowest_gc = min(results, key=lambda r: r["gc_percent"])
    highest_gc = max(results, key=lambda r: r["gc_percent"])
    corr_len_time = float(np.corrcoef([r["genome_len"] for r in results], [r["assembly_time_ms"] for r in results])[0,1]) if len(results)>1 else float("nan")

    report_lines = []
    report_lines.append("Differences Between Point Positions")
    report_lines.append("=" * 34)
    report_lines.append("")
    report_lines.append(f"Number of genomes analyzed: {len(results)}")
    report_lines.append(f"GC% range: {min(xs):.2f}% – {max(xs):.2f}%")
    report_lines.append(f"Assembly time range: {min(ys):.1f} ms – {max(ys):.1f} ms")
    report_lines.append(f"Pearson correlation (GC% vs time): {corr:.3f}")
    report_lines.append(f"Pearson correlation (genome length vs time): {corr_len_time:.3f}")
    report_lines.append("")
    report_lines.append("Notable Points")
    report_lines.append("-" * 14)
    report_lines.append(f"- Fastest (lowest Y): V{fastest['index']:02d} — time {fastest['assembly_time_ms']:.1f} ms, GC {fastest['gc_percent']:.2f}%, length {fastest['genome_len']} bp.")
    report_lines.append(f"- Slowest (highest Y): V{slowest['index']:02d} — time {slowest['assembly_time_ms']:.1f} ms, GC {slowest['gc_percent']:.2f}%, length {slowest['genome_len']} bp.")
    report_lines.append(f"- Lowest GC (leftmost X): V{lowest_gc['index']:02d} — GC {lowest_gc['gc_percent']:.2f}%, time {lowest_gc['assembly_time_ms']:.1f} ms.")
    report_lines.append(f"- Highest GC (rightmost X): V{highest_gc['index']:02d} — GC {highest_gc['gc_percent']:.2f}%, time {highest_gc['assembly_time_ms']:.1f} ms.")
    report_lines.append("")
    report_lines.append("Interpretation")
    report_lines.append("-" * 13)
    report_lines.append(textwrap.fill(
        "Horizontal position (X) reflects overall GC%. Vertical position (Y) reflects the measured time "
        "for our toy de Bruijn assembly. In this toy model, assembly time generally scales with the number "
        "of k-mers, which is influenced by genome length, read count (coverage), and sequence repetitiveness. "
        "GC% per se does not directly cause longer times, but GC-rich or GC-poor sequences can alter k-mer "
        "diversity and repeat structure, indirectly affecting graph complexity.", width=90))
    report_lines.append("")
    report_lines.append(textwrap.fill(
        "When points with similar GC% differ in Y, length and graph complexity likely differ—e.g., longer or "
        "more repetitive genomes produce more nodes/edges and thus take longer in our greedy walk. Conversely, "
        "points with similar Y but different GC% suggest GC% alone is not the main driver here.", width=90))
    report_lines.append("")
    report_lines.append("Table")
    report_lines.append("-" * 5)
    report_lines.append("Label  GC%    Time(ms)  GenomeLen  Nreads  ~N50  #Contigs  Accession/Header")
    report_lines.append("-----  -----  --------  ---------  ------  ----  -------  -----------------")
    for r in results:
        report_lines.append(f"V{r['index']:02d}   {r['gc_percent']:.2f}   {r['assembly_time_ms']:.1f}    {r['genome_len']:>9}  {r['n_reads']:>6}  {r['n50_approx']:>4}  {r['n_contigs']:>7}  {r['ncbi_header']}")

    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text + "\n")

    # Also write a plain .txt file
    report_path = "out/results/explanation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Done.")
    print(f"- Summary JSON: {table_path}")
    print(f"- Text report:  {report_path}")
    print("Outputs are in ./out/ (plot was displayed, not saved)")

if __name__ == "__main__":
    main()
