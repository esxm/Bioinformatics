import math
import tkinter as tk
from tkinter import ttk, messagebox

BASES = "ACGT"


# -----------------------------
# Core computations
# -----------------------------
def normalize_motifs(motifs_raw: str):
    motifs = []
    for line in motifs_raw.splitlines():
        s = line.strip().upper()
        if not s:
            continue
        motifs.append(s)
    if not motifs:
        raise ValueError("No motifs provided.")
    L = len(motifs[0])
    for m in motifs:
        if len(m) != L:
            raise ValueError("All motifs must have the same length.")
        if any(ch not in BASES for ch in m):
            raise ValueError(f"Invalid character in motif '{m}'. Use only A/C/G/T.")
    return motifs, L


def count_matrix(motifs, L):
    counts = {b: [0] * L for b in BASES}
    for m in motifs:
        for i, ch in enumerate(m):
            counts[ch][i] += 1
    return counts


def weight_matrix(counts):
    # As in your slides, "weight matrix" is often just the raw counts.
    return counts


def relative_freq_matrix(counts, pseudocount=0.0):
    # If pseudocount == 0: divide by N exactly (matches the slide table)
    # If pseudocount > 0: add pseudocount to every base in every column
    L = len(next(iter(counts.values())))
    N = sum(counts[b][0] for b in BASES)  # number of motifs
    denom = N + 4.0 * pseudocount
    freqs = {b: [0.0] * L for b in BASES}
    for i in range(L):
        for b in BASES:
            freqs[b][i] = (counts[b][i] + pseudocount) / denom
    return freqs, N


def log_likelihood_matrix(freqs, bg=None):
    if bg is None:
        bg = {b: 0.25 for b in BASES}  # your null model
    L = len(next(iter(freqs.values())))
    ll = {b: [float("-inf")] * L for b in BASES}
    for i in range(L):
        for b in BASES:
            p = freqs[b][i]
            if p <= 0.0:
                ll[b][i] = float("-inf")
            else:
                ll[b][i] = math.log(p / bg[b])
    return ll


def score_window(window, ll):
    s = 0.0
    for i, ch in enumerate(window):
        v = ll[ch][i]
        if v == float("-inf"):
            return float("-inf")
        s += v
    return s


def scan_sequence(seq, L, ll):
    seq = seq.strip().upper()
    if any(ch not in BASES for ch in seq):
        raise ValueError("Sequence S must contain only A/C/G/T.")
    if len(seq) < L:
        raise ValueError(f"Sequence S is shorter than motif length L={L}.")
    out = []
    for i in range(len(seq) - L + 1):
        w = seq[i : i + L]
        out.append((i + 1, w, score_window(w, ll)))  # 1-based position
    return out


# -----------------------------
# Formatting helpers (plain text)
# -----------------------------
def fmt_matrix(mat, title, L, digits=3, inf_as="-inf"):
    # mat: dict base -> list[L]
    lines = []
    lines.append(title)
    header = "     " + " ".join(f"{i:>7d}" for i in range(1, L + 1))
    lines.append(header)
    for b in BASES:
        row = []
        for v in mat[b]:
            if isinstance(v, float):
                if v == float("-inf"):
                    row.append(f"{inf_as:>7}")
                else:
                    row.append(f"{v:>7.{digits}f}")
            else:
                row.append(f"{v:>7d}")
        lines.append(f"{b:>3}  " + " ".join(row))
    lines.append("")
    return "\n".join(lines)


def fmt_scan(scores, top_k=10):
    finite = [x for x in scores if x[2] != float("-inf")]
    finite_sorted = sorted(finite, key=lambda t: t[2], reverse=True)

    lines = []
    lines.append("Sliding-window scores (window length = motif length)")
    lines.append(f"Total windows: {len(scores)}")
    lines.append(f"Finite-score windows: {len(finite)} (others are -inf due to 0-probability positions)")
    lines.append("")

    lines.append("Top hits:")
    if not finite_sorted:
        lines.append("  (none)")
    else:
        for pos, w, sc in finite_sorted[:top_k]:
            lines.append(f"  pos {pos:>3d} : {w}  score = {sc:.6f}")
    lines.append("")

    # Also list all windows (compact)
    lines.append("All windows:")
    for pos, w, sc in scores:
        ssc = "-inf" if sc == float("-inf") else f"{sc:.6f}"
        lines.append(f"  {pos:>3d}  {w}  {ssc}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Tkinter UI
# -----------------------------
DEFAULT_MOTIFS = "\n".join(
    [
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
)

DEFAULT_S = "CAGGTTGGAAACGTAATCAGCGATTACGCATGACGTAA"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DNA Motif Finder (Counts → Log-likelihood → Scan)")
        self.geometry("1100x750")

        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # Left: inputs
        left = ttk.Frame(outer)
        left.pack(side="left", fill="both", expand=False)

        ttk.Label(left, text="Known motif sequences (one per line):").pack(anchor="w")
        self.motifs_text = tk.Text(left, width=35, height=18, font=("Courier", 10))
        self.motifs_text.pack(fill="x", pady=(0, 10))

        ttk.Label(left, text='Sequence S to analyze:').pack(anchor="w")
        self.s_entry = ttk.Entry(left, width=50)
        self.s_entry.pack(fill="x", pady=(0, 10))

        opts = ttk.Frame(left)
        opts.pack(fill="x", pady=(0, 10))

        ttk.Label(opts, text="Pseudocount (0 matches your slide table):").grid(row=0, column=0, sticky="w")
        self.pseudo_var = tk.StringVar(value="0")
        ttk.Entry(opts, textvariable=self.pseudo_var, width=8).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(opts, text="Background (null) model:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.bg_var = tk.StringVar(value="uniform")
        ttk.Radiobutton(opts, text="Uniform (0.25 each)", variable=self.bg_var, value="uniform").grid(
            row=1, column=1, sticky="w", pady=(8, 0)
        )

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Load defaults", command=self._load_defaults).pack(side="left")
        ttk.Button(btns, text="Compute", command=self._compute).pack(side="left", padx=8)

        # Right: output
        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Label(right, text="Output:").pack(anchor="w")
        self.out = tk.Text(right, wrap="none", font=("Courier", 10))
        self.out.pack(fill="both", expand=True)

        # Scrollbars
        yscroll = ttk.Scrollbar(self.out, orient="vertical", command=self.out.yview)
        self.out.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")

        xscroll = ttk.Scrollbar(right, orient="horizontal", command=self.out.xview)
        self.out.configure(xscrollcommand=xscroll.set)
        xscroll.pack(fill="x")

    def _load_defaults(self):
        self.motifs_text.delete("1.0", "end")
        self.motifs_text.insert("1.0", DEFAULT_MOTIFS)
        self.s_entry.delete(0, "end")
        self.s_entry.insert(0, DEFAULT_S)
        self.pseudo_var.set("0")
        self.bg_var.set("uniform")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", "Loaded defaults.\nClick Compute.\n")

    def _compute(self):
        try:
            motifs_raw = self.motifs_text.get("1.0", "end")
            motifs, L = normalize_motifs(motifs_raw)

            try:
                pseudocount = float(self.pseudo_var.get().strip())
                if pseudocount < 0:
                    raise ValueError
            except ValueError:
                raise ValueError("Pseudocount must be a non-negative number (e.g., 0 or 0.5 or 1).")

            # Null model
            if self.bg_var.get() == "uniform":
                bg = {b: 0.25 for b in BASES}
            else:
                bg = {b: 0.25 for b in BASES}

            counts = count_matrix(motifs, L)
            weights = weight_matrix(counts)
            freqs, N = relative_freq_matrix(counts, pseudocount=pseudocount)
            ll = log_likelihood_matrix(freqs, bg=bg)

            S = self.s_entry.get().strip().upper()
            scores = scan_sequence(S, L, ll)

            # Simple "signal" heuristic: does S contain a high-scoring hit?
            finite = [x for x in scores if x[2] != float("-inf")]
            best = max(finite, key=lambda t: t[2]) if finite else None

            # Compute theoretical max (best base per position)
            max_possible = 0.0
            for i in range(L):
                max_possible += max(ll[b][i] for b in BASES)

            # Write output
            self.out.delete("1.0", "end")
            self.out.insert("end", f"Motifs: N={N}, motif length L={L}\n")
            self.out.insert("end", f"Pseudocount={pseudocount}  |  Background: uniform 0.25 each\n\n")

            self.out.insert("end", fmt_matrix(counts, "1) Count matrix", L))
            self.out.insert("end", fmt_matrix(weights, "2) Weight matrix (same as counts here)", L))
            self.out.insert("end", fmt_matrix(freqs, "3) Relative frequencies matrix P(b,i)", L, digits=3))
            self.out.insert("end", fmt_matrix(ll, "4) Log-likelihoods matrix ln(P(b,i)/0.25)", L, digits=6))

            self.out.insert("end", "5) Scan S with sliding window\n")
            self.out.insert("end", f"S length = {len(S)} ; windows = {len(scores)}\n")
            self.out.insert("end", f"Theoretical max score (best base per column) = {max_possible:.6f}\n\n")
            self.out.insert("end", fmt_scan(scores, top_k=10))

            self.out.insert("end", "Conclusion / signal check:\n")
            if best is None:
                self.out.insert("end", "  No finite-scoring windows found (likely due to zero probabilities).\n")
                self.out.insert("end", "  Try adding a pseudocount (e.g., 0.5 or 1) to avoid -inf.\n")
            else:
                pos, w, sc = best
                frac = sc / max_possible if max_possible != 0 else 0.0
                self.out.insert("end", f"  Best hit at position {pos}: {w} with score {sc:.6f}\n")
                self.out.insert("end", f"  This is {frac*100:.1f}% of the theoretical maximum.\n")
                if sc > 0:
                    self.out.insert(
                        "end",
                        "  ✅ Signal present: a positive log-likelihood hit suggests a motif-like exon–intron border pattern.\n",
                    )
                else:
                    self.out.insert(
                        "end",
                        "  Weak/negative signal: best hit does not exceed background.\n",
                    )

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    App().mainloop()
