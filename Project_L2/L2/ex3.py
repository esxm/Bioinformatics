import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("TkAgg")  # Embed in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------
# FASTA parsing utilities
# --------------------------

def read_fasta(filepath: str) -> str:
    """
    Read a FASTA file (single or multi-record). Concatenate all sequences into one.
    Ignores header lines starting with '>'.
    Returns uppercase sequence string with whitespace removed.
    """
    seq_parts = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line)
    seq = "".join(seq_parts).replace(" ", "").upper()
    return seq

# --------------------------
# Sliding window frequencies
# --------------------------

def sliding_window_relative_freqs(
    seq: str,
    window: int = 30,
    alphabet: Tuple[str, ...] = ("A", "C", "G", "T"),
    exclude_non_alphabet_from_denominator: bool = True,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    Compute relative frequency vectors for each symbol in `alphabet`
    over sliding windows across `seq`.

    Parameters
    ----------
    seq : str
        Sequence to analyze (uppercase recommended).
    window : int
        Window size (default 30).
    alphabet : tuple[str]
        Symbols to track (default DNA A,C,G,T).
    exclude_non_alphabet_from_denominator : bool
        If True, the relative frequency denominator is the count of
        characters within the window that are in `alphabet`.
        If False, denominator is the full window length.

    Returns
    -------
    positions : list[int]
        1-based start positions of each window.
    freqs : dict[str, list[float]]
        For each symbol, a list of relative frequencies per window.
    """
    n = len(seq)
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if n < window:
        return [], {sym: [] for sym in alphabet}

    positions = []
    freqs = {sym: [] for sym in alphabet}

    for start in range(0, n - window + 1):
        win = seq[start:start + window]
        if exclude_non_alphabet_from_denominator:
            denom = sum(1 for ch in win if ch in alphabet)
        else:
            denom = window

        counts = {sym: 0 for sym in alphabet}
        for ch in win:
            if ch in counts:
                counts[ch] += 1

        # Avoid division by zero (e.g., if window has 0 valid symbols)
        if denom == 0:
            rels = {sym: 0.0 for sym in alphabet}
        else:
            rels = {sym: counts[sym] / denom for sym in alphabet}

        positions.append(start + 1)  # 1-based indexing for display
        for sym in alphabet:
            freqs[sym].append(rels[sym])

    return positions, freqs

# --------------------------
# GUI Application
# --------------------------

class FastaFreqApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FASTA Sliding-Window Nucleotide Frequencies")
        self.geometry("1000x700")

        self.seq = ""
        self.filepath = None

        self._build_ui()

    def _build_ui(self):
        # Top controls frame
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # Open button
        self.open_btn = ttk.Button(controls, text="Open FASTA…", command=self.open_fasta)
        self.open_btn.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        # Selected file label
        self.file_label_var = tk.StringVar(value="No file selected")
        self.file_label = ttk.Label(controls, textvariable=self.file_label_var, width=60)
        self.file_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        # Window size
        ttk.Label(controls, text="Window size:").grid(row=0, column=2, padx=(20,5), pady=2, sticky="e")
        self.window_var = tk.IntVar(value=30)
        self.window_entry = ttk.Spinbox(controls, from_=1, to=10000, textvariable=self.window_var, width=8)
        self.window_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")

        # Denominator choice
        self.exclude_non_alpha_var = tk.BooleanVar(value=True)
        self.exclude_chk = ttk.Checkbutton(
            controls,
            text="Exclude non-ACGT from denominator",
            variable=self.exclude_non_alpha_var
        )
        self.exclude_chk.grid(row=0, column=4, padx=(20,5), pady=2, sticky="w")

        # Analyze button
        self.run_btn = ttk.Button(controls, text="Analyze & Plot", command=self.analyze_and_plot)
        self.run_btn.grid(row=0, column=5, padx=15, pady=2, sticky="e")

        # Info box
        self.info_var = tk.StringVar(value="Load a FASTA file to begin.")
        info = ttk.Label(self, textvariable=self.info_var, foreground="#333")
        info.pack(side=tk.TOP, anchor="w", padx=12, pady=(0,8))

        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Sliding-Window Relative Frequencies (A, C, G, T)")
        self.ax.set_xlabel("Window start position (1-based)")
        self.ax.set_ylabel("Relative frequency")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # Style cleanups
        try:
            self.tk.call("source", "sun-valley.tcl")
            self.tk.call("set_theme", "light")
        except Exception:
            pass  # theme optional

    def open_fasta(self):
        filepath = filedialog.askopenfilename(
            title="Select FASTA file",
            filetypes=[("FASTA files", "*.fa *.fasta *.fna *.ffn *.faa *.frn"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            seq = read_fasta(filepath)
            if not seq:
                raise ValueError("No sequence data found in the file.")
            self.seq = seq
            self.filepath = filepath
            self.file_label_var.set(os.path.basename(filepath))
            self.info_var.set(f"Loaded sequence length: {len(self.seq)} | Alphabet encountered: {''.join(sorted(set(self.seq)))}")
            self.status_var.set("File loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error reading FASTA", str(e))
            self.status_var.set("Failed to load file.")

    def analyze_and_plot(self):
        if not self.seq:
            messagebox.showwarning("No sequence", "Please open a FASTA file first.")
            return

        try:
            window = int(self.window_var.get())
            if window <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid window size", "Window size must be a positive integer.")
            return

        exclude_non_alpha = bool(self.exclude_non_alpha_var.get())

        # Fixed 4-symbol alphabet per requirement
        alphabet = ("A", "C", "G", "T")

        # Compute frequency vectors
        positions, freqs = sliding_window_relative_freqs(
            self.seq,
            window=window,
            alphabet=alphabet,
            exclude_non_alphabet_from_denominator=exclude_non_alpha,
        )

        # Update info
        if not positions:
            self.info_var.set(
                f"Sequence length ({len(self.seq)}) is smaller than the window size ({window})."
            )
            self._clear_plot()
            return

        total_windows = len(positions)
        self.info_var.set(
            f"Analyzed {total_windows} windows | Window size={window} | "
            f"Denominator={'valid ACGT only' if exclude_non_alpha else 'full window length'}"
        )

        # Plot
        self._plot_freqs(positions, freqs, alphabet)

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Sliding-Window Relative Frequencies (A, C, G, T)")
        self.ax.set_xlabel("Window start position (1-based)")
        self.ax.set_ylabel("Relative frequency")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.canvas.draw_idle()

    def _plot_freqs(self, positions: List[int], freqs: Dict[str, List[float]], alphabet: Tuple[str, ...]):
        self.ax.clear()
        self.ax.set_title("Sliding-Window Relative Frequencies (A, C, G, T)")
        self.ax.set_xlabel("Window start position (1-based)")
        self.ax.set_ylabel("Relative frequency")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # Plot one line per symbol (order A,C,G,T)
        for sym in alphabet:
            self.ax.plot(positions, freqs[sym], label=sym, linewidth=1.6)

        self.ax.legend(title="Nucleotide", ncol=len(alphabet), frameon=True)
        self.canvas.draw_idle()

def main():
    app = FastaFreqApp()
    app.mainloop()

if __name__ == "__main__":
    main()
