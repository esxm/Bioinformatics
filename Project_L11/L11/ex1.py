import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np

# Default sequences
DEFAULT_S1 = "ACCGTGAAGCCAATAC"
DEFAULT_S2 = "AGCGTGCAGCCAATAC"

# Default scoring scheme (used only to prefill the GUI)
DEFAULT_MATCH = 1
DEFAULT_MISMATCH = -1
DEFAULT_GAP = 0  # screenshot shows gap = 0 by default


def needleman_wunsch(seq1, seq2, match_score, mismatch_score, gap_penalty):
    n = len(seq1)
    m = len(seq2)

    # Score and traceback matrices
    score = [[0] * (m + 1) for _ in range(n + 1)]
    traceback = [[None] * (m + 1) for _ in range(n + 1)]

    # Initialize first row/column
    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
        traceback[i][0] = "up"
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
        traceback[0][j] = "left"

    # Fill matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score[i - 1][j - 1] + match_score
            else:
                diag_score = score[i - 1][j - 1] + mismatch_score

            up_score = score[i - 1][j] + gap_penalty
            left_score = score[i][j - 1] + gap_penalty

            max_score = max(diag_score, up_score, left_score)
            score[i][j] = max_score

            if max_score == diag_score:
                traceback[i][j] = "diag"
            elif max_score == up_score:
                traceback[i][j] = "up"
            else:
                traceback[i][j] = "left"

    # Traceback to get alignment and path
    aligned1 = []
    aligned2 = []
    i, j = n, m
    path = [(i, j)]

    while i > 0 or j > 0:
        direction = traceback[i][j]
        if direction == "diag":
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif direction == "up":
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1
        elif direction == "left":
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1
        else:
            break
        path.append((i, j))

    aligned1 = ''.join(reversed(aligned1))
    aligned2 = ''.join(reversed(aligned2))
    path = list(reversed(path))

    return aligned1, aligned2, score, path


class NWGuiApp:
    def __init__(self, master):
        self.master = master
        master.title("Needleman–Wunsch DNA Alignment")

        # ====== Top frame: sequences, params, and button ======
        top_frame = ttk.Frame(master, padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Left: sequences
        seq_frame = ttk.Frame(top_frame)
        seq_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ttk.Label(seq_frame, text="Sequence 1:").grid(row=0, column=0, sticky="w")
        self.seq1_entry = ttk.Entry(seq_frame, width=40)
        self.seq1_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(seq_frame, text="Sequence 2:").grid(row=1, column=0, sticky="w")
        self.seq2_entry = ttk.Entry(seq_frame, width=40)
        self.seq2_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        self.seq1_entry.insert(0, DEFAULT_S1)
        self.seq2_entry.insert(0, DEFAULT_S2)

        # Middle: editable parameters
        params_frame = ttk.LabelFrame(top_frame, text="Parameters")
        params_frame.grid(row=0, column=1, padx=15, pady=5, sticky="n")

        ttk.Label(params_frame, text="Match:").grid(row=0, column=0, sticky="e")
        self.e_match = ttk.Entry(params_frame, width=8)
        self.e_match.grid(row=0, column=1, padx=3, pady=2)
        self.e_match.insert(0, str(DEFAULT_MATCH))

        ttk.Label(params_frame, text="Mismatch:").grid(row=1, column=0, sticky="e")
        self.e_mismatch = ttk.Entry(params_frame, width=8)
        self.e_mismatch.grid(row=1, column=1, padx=3, pady=2)
        self.e_mismatch.insert(0, str(DEFAULT_MISMATCH))

        ttk.Label(params_frame, text="Gap:").grid(row=2, column=0, sticky="e")
        self.e_gap = ttk.Entry(params_frame, width=8)
        self.e_gap.grid(row=2, column=1, padx=3, pady=2)
        self.e_gap.insert(0, str(DEFAULT_GAP))

        # Right: Align button
        self.align_button = ttk.Button(top_frame, text="Align", command=self.run_alignment)
        self.align_button.grid(row=0, column=2, padx=10, pady=5, sticky="n")

        # ====== Middle frame: plot(s) and text ======
        middle_frame = ttk.Frame(master, padding=5)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left side: two plots (score matrix + deviation)
        plots_frame = ttk.Frame(middle_frame)
        plots_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Score matrix + path
        self.fig_score = Figure(figsize=(4, 4), dpi=100)
        self.ax_score = self.fig_score.add_subplot(111)
        self.canvas_score = FigureCanvasTkAgg(self.fig_score, master=plots_frame)
        self.canvas_score_widget = self.canvas_score.get_tk_widget()
        self.canvas_score_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Deviation-from-diagonal plot
        self.fig_dev = Figure(figsize=(4, 4), dpi=100)
        self.ax_dev = self.fig_dev.add_subplot(111)
        self.canvas_dev = FigureCanvasTkAgg(self.fig_dev, master=plots_frame)
        self.canvas_dev_widget = self.canvas_dev.get_tk_widget()
        self.canvas_dev_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: text output
        right_frame = ttk.Frame(middle_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="Alignment:").pack(anchor="w")
        self.text = tk.Text(right_frame, width=50, height=20)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Initial run
        self.run_alignment()

    def get_params(self):
        """Read parameters from GUI; fall back to defaults if parsing fails."""
        try:
            match = int(self.e_match.get())
        except ValueError:
            match = DEFAULT_MATCH

        try:
            mismatch = int(self.e_mismatch.get())
        except ValueError:
            mismatch = DEFAULT_MISMATCH

        try:
            gap = int(self.e_gap.get())
        except ValueError:
            gap = DEFAULT_GAP

        return match, mismatch, gap

    def run_alignment(self):
        seq1 = self.seq1_entry.get().strip().upper()
        seq2 = self.seq2_entry.get().strip().upper()
        if not seq1 or not seq2:
            return

        match, mismatch, gap = self.get_params()

        aligned1, aligned2, score_matrix, path = needleman_wunsch(
            seq1, seq2, match, mismatch, gap
        )

        # Update text output
        self.text.delete("1.0", tk.END)

        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b)
        length = len(aligned1)
        similarity = matches / length * 100 if length > 0 else 0
        final_score = score_matrix[len(seq1)][len(seq2)]

        match_line = ''.join('|' if a == b else ' ' for a, b in zip(aligned1, aligned2))

        self.text.insert(tk.END, f"Seq1: {aligned1}\n")
        self.text.insert(tk.END, f"      {match_line}\n")
        self.text.insert(tk.END, f"Seq2: {aligned2}\n\n")
        self.text.insert(tk.END, f"Matches: {matches}\n")
        self.text.insert(tk.END, f"Length: {length}\n")
        self.text.insert(tk.END, f"Similarity: {similarity:.1f} %\n")
        self.text.insert(tk.END, f"Final alignment score: {final_score}\n")
        self.text.insert(
            tk.END,
            f"Scoring: match={match}, mismatch={mismatch}, gap={gap}\n"
        )

        # Update plots
        self.update_plots(score_matrix, path, seq1, seq2)

    def update_plots(self, score_matrix, path, seq1, seq2):
        n = len(seq1)
        m = len(seq2)

        # ===== Score matrix plot (left) =====
        self.ax_score.clear()

        # reddish colormap similar to original app
        self.ax_score.imshow(score_matrix, origin='upper', aspect='auto', cmap='magma')

        xs = [j for (i, j) in path]
        ys = [i for (i, j) in path]
        # show path as squares instead of a thin line
        self.ax_score.plot(xs, ys, linestyle='-', linewidth=2, color='cyan')

        self.ax_score.set_xticks(range(m + 1))
        self.ax_score.set_yticks(range(n + 1))
        self.ax_score.set_xticklabels(['-'] + list(seq2))
        self.ax_score.set_yticklabels(['-'] + list(seq1))
        self.ax_score.set_xlabel("Sequence 2")
        self.ax_score.set_ylabel("Sequence 1")
        self.ax_score.set_title("Score matrix & optimal path")

        self.fig_score.tight_layout()
        self.canvas_score.draw_idle()

        # ===== Deviation-from-diagonal plot (right) =====
        self.ax_dev.clear()

        # Matrix for path: 0 = background, 1 = cells on traceback path
        dev = np.zeros((n + 1, m + 1), dtype=int)
        for i, j in path:
            dev[i, j] = 1

        # Yellow background, red path (like original)
        cmap = ListedColormap(["#fff9c4", "#d32f2f"])  # pale yellow, dark red
        self.ax_dev.imshow(dev, origin='upper', aspect='equal', cmap=cmap, vmin=0, vmax=1)

        # Draw grid
        self.ax_dev.set_xticks(np.arange(-0.5, m + 1, 1), minor=True)
        self.ax_dev.set_yticks(np.arange(-0.5, n + 1, 1), minor=True)
        self.ax_dev.grid(which='minor', linestyle='-', linewidth=0.5, color='black')

        # No tick labels (matches screenshot look)
        self.ax_dev.set_xticks([])
        self.ax_dev.set_yticks([])

        self.ax_dev.set_title("Traceback path deviation\nfrom optimal alignment (diagonal)")

        self.fig_dev.tight_layout()
        self.canvas_dev.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = NWGuiApp(root)
    root.mainloop()
