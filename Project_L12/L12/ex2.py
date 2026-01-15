import math
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple

import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# HARD-CODED SETTINGS
# =========================
EMAIL = "mmswflow@gmail.com"
API_KEY = None

N_GENOMES = 10
NCBI_QUERY = 'Influenza A virus[Organism] AND "complete genome"[Title]'

PSEUDOCOUNT = 1.0
BACKGROUND_MODEL = "genome"  # "genome" or "uniform"

TOP_K_PEAKS = 5
MIN_DISTANCE_BETWEEN_PEAKS = 200

MOTIFS = [
    "GAGGTAAAC",
    "TCCGTAAGT",
    "CAGGTTGGA",
    "ACAGTCAGT",
    "TAGGTCATT",
    "TAGGTACTG",
    "ATGGTAACT",
    "CAGGTATAC",
    "TGTGTGAGT",
    "AAGGTAAGT",
]

BASES = "ACGT"


# -----------------------------
# FASTA parsing
# -----------------------------
@dataclass
class FastaRecord:
    header: str
    seq: str


def parse_fasta(text: str) -> List[FastaRecord]:
    records = []
    header = None
    seq_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(FastaRecord(header=header, seq="".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        records.append(FastaRecord(header=header, seq="".join(seq_lines)))
    return records


def normalize_sequence_keep_coords(seq: str) -> str:
    s = seq.upper().replace("U", "T")
    s = re.sub(r"[^ACGT]", "N", s)
    return s


# -----------------------------
# NCBI E-utilities
# -----------------------------
def ncbi_esearch(query: str, retmax: int) -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "nuccore",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "email": EMAIL,
        "tool": "influenza_motif_gui",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def ncbi_efetch_fasta(id_list: List[str]) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": ",".join(id_list),
        "rettype": "fasta",
        "retmode": "text",
        "email": EMAIL,
        "tool": "influenza_motif_gui",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    return r.text


# -----------------------------
# PWM / scoring
# -----------------------------
def normalize_motifs(motifs: List[str]) -> Tuple[List[str], int]:
    motifs = [m.strip().upper() for m in motifs if m.strip()]
    if not motifs:
        raise ValueError("No motifs provided.")
    L = len(motifs[0])
    for m in motifs:
        if len(m) != L:
            raise ValueError("All motifs must have the same length.")
        if any(ch not in BASES for ch in m):
            raise ValueError(f"Motif contains invalid char: {m}")
    return motifs, L


def count_matrix(motifs: List[str], L: int) -> Dict[str, List[int]]:
    counts = {b: [0] * L for b in BASES}
    for m in motifs:
        for i, ch in enumerate(m):
            counts[ch][i] += 1
    return counts


def relative_freq_matrix(counts: Dict[str, List[int]], pseudocount: float) -> Tuple[Dict[str, List[float]], int]:
    L = len(next(iter(counts.values())))
    N = sum(counts[b][0] for b in BASES)
    denom = N + 4.0 * pseudocount
    freqs = {b: [0.0] * L for b in BASES}
    for i in range(L):
        for b in BASES:
            freqs[b][i] = (counts[b][i] + pseudocount) / denom
    return freqs, N


def background_from_sequence(seq: str) -> Dict[str, float]:
    c = {b: 0 for b in BASES}
    for ch in seq:
        if ch in c:
            c[ch] += 1
    total = sum(c.values())
    if total == 0:
        return {b: 0.25 for b in BASES}
    return {b: c[b] / total for b in BASES}


def log_likelihood_matrix(freqs: Dict[str, List[float]], bg: Dict[str, float]) -> Dict[str, List[float]]:
    L = len(next(iter(freqs.values())))
    ll = {b: [float("-inf")] * L for b in BASES}
    for i in range(L):
        for b in BASES:
            p = freqs[b][i]
            q = bg.get(b, 0.0)
            if p <= 0.0 or q <= 0.0:
                ll[b][i] = float("-inf")
            else:
                ll[b][i] = math.log(p / q)
    return ll


def score_window(window: str, ll: Dict[str, List[float]]) -> float:
    s = 0.0
    for i, ch in enumerate(window):
        if ch not in BASES:
            return float("-inf")
        v = ll[ch][i]
        if v == float("-inf"):
            return float("-inf")
        s += v
    return s


def scan_sequence(seq: str, L: int, ll: Dict[str, List[float]]) -> List[float]:
    return [score_window(seq[i : i + L], ll) for i in range(0, len(seq) - L + 1)]


def local_maxima(scores: List[float], min_distance: int, top_k: int) -> List[Tuple[int, float]]:
    cand = []
    for i in range(1, len(scores) - 1):
        if scores[i] != float("-inf") and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            cand.append((i, scores[i]))
    cand.sort(key=lambda x: x[1], reverse=True)

    picked = []
    for i0, sc in cand:
        if all(abs(i0 - p[0]) >= min_distance for p in picked):
            picked.append((i0, sc))
        if len(picked) >= top_k:
            break

    return [(i + 1, sc) for i, sc in picked]  # 1-based window start


def accession_from_header(header: str) -> str:
    return header.split()[0] if header else ""


# -----------------------------
# Tkinter UI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Influenza Motif Scanner (Fixed UI)")
        self.geometry("1250x820")

        # FIX: pack a top bar FIRST, then the notebook. No place().
        self.topbar = ttk.Frame(self, padding=(10, 8))
        self.topbar.pack(side="top", fill="x")

        self.run_btn = ttk.Button(self.topbar, text="Download & Scan 10 Genomes", command=self.start_scan)
        self.run_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.topbar, textvariable=self.status_var).pack(side="left", padx=12)

        ttk.Separator(self, orient="horizontal").pack(side="top", fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True)

        # Summary tab
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Summary")

        self.tree = ttk.Treeview(
            self.summary_tab,
            columns=("idx", "accession", "len", "best_pos", "best_score", "peaks"),
            show="headings",
            height=20,
        )
        for col, txt, w in [
            ("idx", "#", 50),
            ("accession", "Accession", 150),
            ("len", "Length", 90),
            ("best_pos", "Best peak pos", 120),
            ("best_score", "Best score", 110),
            ("peaks", "Top peaks (pos:score)", 750),
        ]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Precompute PWM parts
        motifs, self.L = normalize_motifs(MOTIFS)
        counts = count_matrix(motifs, self.L)
        self.freqs, self.N_motifs = relative_freq_matrix(counts, PSEUDOCOUNT)

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        while len(self.notebook.tabs()) > 1:
            self.notebook.forget(1)

    def start_scan(self):
        self.run_btn.config(state="disabled")
        self.status_var.set("Downloading from NCBI and scanning...")
        self.clear_results()
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            ids = ncbi_esearch(NCBI_QUERY, retmax=N_GENOMES)
            if not ids:
                raise RuntimeError("NCBI returned no IDs for the query.")

            fasta_text = ncbi_efetch_fasta(ids)
            records = parse_fasta(fasta_text)
            if not records:
                raise RuntimeError("Failed to parse FASTA from NCBI response.")
            records = records[:N_GENOMES]

            results = []
            for idx, rec in enumerate(records, start=1):
                seq = normalize_sequence_keep_coords(rec.seq)
                if len(seq) < self.L:
                    results.append((idx, rec, None, None, [], (None, None)))
                    continue

                bg = {b: 0.25 for b in BASES} if BACKGROUND_MODEL == "uniform" else background_from_sequence(seq)
                ll = log_likelihood_matrix(self.freqs, bg=bg)
                scores = scan_sequence(seq, self.L, ll)
                peaks = local_maxima(scores, MIN_DISTANCE_BETWEEN_PEAKS, TOP_K_PEAKS)
                best = peaks[0] if peaks else (None, None)
                results.append((idx, rec, seq, scores, peaks, best))

            self.after(0, lambda: self._render_results(results))

        except Exception as e:
            self.after(0, lambda: self._show_error(str(e)))

    def _show_error(self, msg: str):
        self.run_btn.config(state="normal")
        self.status_var.set("Error.")
        messagebox.showerror("Error", msg)

    def _render_results(self, results):
        # Summary
        for idx, rec, seq, scores, peaks, best in results:
            acc = accession_from_header(rec.header)
            length = len(seq) if seq else 0
            best_pos, best_score = best
            peaks_str = "; ".join([f"{p}:{s:.3f}" for p, s in peaks]) if peaks else ""

            self.tree.insert(
                "",
                "end",
                values=(
                    idx,
                    acc,
                    length,
                    best_pos if best_pos is not None else "",
                    f"{best_score:.3f}" if isinstance(best_score, (int, float)) else "",
                    peaks_str,
                ),
            )

        # Plot tabs
        for idx, rec, seq, scores, peaks, best in results:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"{idx:02d} {accession_from_header(rec.header)}")

            info = ttk.Label(
                tab,
                text=(
                    f"Header: {rec.header}\n"
                    f"Length: {len(seq) if seq else 0} | Motif length L={self.L} | N_motifs={self.N_motifs} "
                    f"| bg={BACKGROUND_MODEL} | pseudocount={PSEUDOCOUNT}\n"
                    f"Best peak: {best[0]}  score={best[1]}"
                ),
                justify="left",
                wraplength=1180,
            )
            info.pack(anchor="w", padx=10, pady=(10, 6))

            fig = Figure(figsize=(11.8, 3.8), dpi=100)
            ax = fig.add_subplot(111)

            if not scores:
                ax.text(0.5, 0.5, "No scores (sequence too short).", ha="center", va="center")
                ax.set_axis_off()
            else:
                xs = list(range(1, len(scores) + 1))
                ys = [float("nan") if v == float("-inf") else v for v in scores]
                ax.plot(xs, ys)
                ax.set_xlabel("Window start position (1-based)")
                ax.set_ylabel("Log-likelihood score (log-odds)")
                ax.set_title("Motif score signal across genome")

                if peaks:
                    px = [p for p, _ in peaks]
                    py = [s for _, s in peaks]
                    ax.scatter(px, py)
                    for p, s in peaks:
                        ax.annotate(str(p), (p, s), textcoords="offset points", xytext=(0, 8), ha="center")

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.run_btn.config(state="normal")
        self.status_var.set("Done. Results displayed in tabs.")
        self.notebook.select(self.summary_tab)


if __name__ == "__main__":
    App().mainloop()
