#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import random
from io import StringIO
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ------------------------------
# Configuration
# ------------------------------
EMAIL = "mohamedmariosakka@gmail.com"  # used for NCBI Entrez
OUT_DIR = "ecoRI_influenza_figs_only"
N_GENOMES = 10
SITE = "GAATTC"        # EcoRI recognition sequence; cut G^AATTC
TOTAL_LEN = 13500      # fallback synthetic length (~influenza total cDNA length)

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(123)
np.random.seed(123)

# ------------------------------
# NCBI: download 10 influenza genomes (DNA; convert U->T if present)
# ------------------------------
def fetch_influenza_genomes_via_entrez(n: int, email: str) -> Dict[str, str]:
    """Return up to n influenza genomes {name: DNA_seq} from NCBI nuccore."""
    from Bio import Entrez, SeqIO
    Entrez.email = email

    # Query for complete genomes from Influenza A/B (broad, but limited to n)
    query = '("Influenza A virus"[Organism] OR "Influenza B virus"[Organism]) AND "complete genome"[Title]'
    handle = Entrez.esearch(db="nuccore", term=query, retmax=n, sort="relevance")
    rec = Entrez.read(handle)
    handle.close()

    ids = rec.get("IdList", [])[:n]
    if not ids:
        return {}

    handle = Entrez.efetch(db="nuccore", id=",".join(ids), rettype="fasta", retmode="text")
    fasta = handle.read()
    handle.close()

    from Bio import SeqIO as _SeqIO
    seqs = {}
    for r in _SeqIO.parse(StringIO(fasta), "fasta"):
        name = r.id
        seq = str(r.seq).upper().replace("U", "T")
        seqs[name] = seq
        if len(seqs) >= n:
            break
    return seqs

# ------------------------------
# Fallback synthetic genomes (only used if download fails)
# ------------------------------
def random_dna(length: int, p_gc: float = 0.45) -> str:
    at = (1.0 - p_gc) / 2.0
    gc = p_gc / 2.0
    bases = (['A'] * int(at * 100) + ['T'] * int(at * 100) +
             ['G'] * int(gc * 100) + ['C'] * int(gc * 100))
    return ''.join(random.choice(bases) for _ in range(length))

def sprinkle_sites(seq: str, site: str, every_k: int = 4000, jitter: int = 1200) -> str:
    s = list(seq)
    pos = max(200, int(random.gauss(every_k, jitter)))
    while pos < len(s) - len(site) - 200:
        s[pos:pos+len(site)] = list(site)
        pos += max(500, int(random.gauss(every_k, jitter)))
    return ''.join(s)

def synthesize_genomes(n: int) -> Dict[str, str]:
    genomes = {}
    for i in range(n):
        seq = random_dna(TOTAL_LEN, p_gc=0.45 + np.random.uniform(-0.03, 0.03))
        seq = sprinkle_sites(seq, SITE, every_k=4000, jitter=1200)
        genomes[f"SYN_Genome_{i+1}"] = seq
    return genomes

# ------------------------------
# Restriction digest + gel utilities
# ------------------------------
def digest(seq: str, site: str) -> List[int]:
    """Return fragment sizes after EcoRI digest (cut between G^AATTC, after the G)."""
    cuts = []
    i = seq.find(site)
    while i != -1:
        cuts.append(i + 1)  # cut after 'G'
        i = seq.find(site, i + 1)
    positions = [0] + sorted(cuts) + [len(seq)]
    return [positions[i+1] - positions[i] for i in range(len(positions)-1)]

def ladder_sizes() -> List[int]:
    return [10000, 8000, 6000, 5000, 4000, 3000, 2500, 2000, 1500,
            1200, 1000, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]

def band_pos(bp: int, gel_h: int, min_bp: int = 100, max_bp: int = 15000) -> int:
    # log-scale mapping; larger fragments remain near the top
    bp = max(min_bp, min(max_bp, bp))
    y = (math.log10(bp) - math.log10(min_bp)) / (math.log10(max_bp) - math.log10(min_bp))
    y = 1.0 - y
    return int(y * (gel_h - 1))

def render_lane(frags: List[int], lane_w: int, gel_h: int, intensity_scale: float = 0.85) -> Image.Image:
    lane = Image.new("L", (lane_w, gel_h), color=0)
    draw = ImageDraw.Draw(lane)
    for f in frags:
        y = band_pos(f, gel_h)
        thickness = max(1, int(2 + 4 * (1 - y / gel_h)))
        intensity = int(255 * min(1.0, intensity_scale * (1 / max(10.0, math.sqrt(max(100, f)))) * 300))
        draw.rectangle([0, max(0, y - thickness//2), lane_w-1, min(gel_h-1, y + thickness//2)], fill=intensity)
    return lane

def compose_gel(lanes: List[List[int]], labels: List[str],
                gel_h: int = 900, lane_w: int = 60, spacer: int = 18,
                include_ladder: bool = True) -> Image.Image:
    lanes_draw = list(lanes)
    labels_draw = list(labels)
    if include_ladder:
        lanes_draw = [ladder_sizes()] + lanes_draw
        labels_draw = ["Ladder"] + labels_draw
    nlanes = len(lanes_draw)
    width = nlanes * lane_w + (nlanes + 1) * spacer + 140
    gel = Image.new("L", (width, gel_h), color=0)
    draw = ImageDraw.Draw(gel)
    x = spacer
    for fraglist, label in zip(lanes_draw, labels_draw):
        lane_img = render_lane(fraglist, lane_w, gel_h)
        gel.paste(lane_img, (x, 0))
        # vertical label at bottom
        lbl = Image.new("L", (lane_w, 16), 0)
        ImageDraw.Draw(lbl).text((2, 0), label[:12], fill=255)
        lbl = lbl.rotate(90, expand=1)
        gel.paste(lbl, (x + lane_w//2 - 8, gel_h - lbl.height - 4))
        x += lane_w + spacer
    # ladder ticks on the right
    for s in ladder_sizes():
        y = band_pos(s, gel_h)
        draw.text((width - 130, y-6), f"{s:>5} bp", fill=200)
        draw.line([(width - 150, y), (width - 140, y)], fill=180, width=1)
    return gel

# ------------------------------
# Main (ONLY the two required figures)
# ------------------------------
def main():
    # Try real downloads; if they fail, synthesize same-count genomes.
    genomes = {}
    try:
        genomes = fetch_influenza_genomes_via_entrez(N_GENOMES, EMAIL)
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
    if not genomes:
        print("[INFO] Using synthetic influenza-like genomes (offline fallback).")
        genomes = synthesize_genomes(N_GENOMES)

    # EcoRI digests
    digests: Dict[str, List[int]] = {name: digest(seq, SITE) for name, seq in genomes.items()}

    # Which genome has the most bands?
    bands = {k: len(v) for k, v in digests.items()}
    top_name = max(bands, key=bands.get)
    top_bands = bands[top_name]
    print(f"[RESULT] Most bands: {top_name} ({top_bands} fragments)")

    # FIGURE 1 — Combined electrophoresis gel (comparison across 10 genomes)
    labels = list(digests.keys())
    lanes = [digests[k] for k in labels]
    gel = compose_gel(lanes, labels, gel_h=900, lane_w=60, spacer=18, include_ladder=True)
    combined_gel_path = os.path.join(OUT_DIR, "combined_gel.png")
    gel.save(combined_gel_path)
    print(f"[OK] Saved combined gel: {combined_gel_path}")

    # FIGURE 2 — Overlay plot (same graph, different colors) of fragment sizes
    # For comparability, plot each genome's fragment sizes sorted desc vs index.
    plt.figure(figsize=(10, 6))
    for name in labels:
        sizes = sorted(digests[name], reverse=True)
        plt.plot(range(1, len(sizes)+1), sizes, marker='o', linewidth=1.5, label=name)  # different colors via Matplotlib
    plt.xlabel("Fragment index (largest → smallest)")
    plt.ylabel("Fragment size (bp)")
    plt.title(f"EcoRI digest fragment sizes — most bands: {top_name} ({top_bands})")
    plt.legend(ncol=2, fontsize=8)
    overlay_path = os.path.join(OUT_DIR, "overlay_fragments.png")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()
    print(f"[OK] Saved overlay plot: {overlay_path}")

if __name__ == "__main__":
    main()
