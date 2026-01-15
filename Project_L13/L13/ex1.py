import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple
import math

EPS = 1e-9


def parse_matrix(text: str, n: int) -> List[List[float]]:
    """
    Parse an NxN matrix from text.
    Accepts space/comma separated values, one row per line.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != n:
        raise ValueError(f"Expected {n} rows, got {len(lines)}.")

    matrix = []
    for i, ln in enumerate(lines):
        parts = [p for p in ln.replace(",", " ").split() if p]
        if len(parts) != n:
            raise ValueError(f"Row {i+1}: expected {n} values, got {len(parts)}.")
        row = [float(x) for x in parts]
        matrix.append(row)
    return matrix


def parse_vector(text: str, n: int) -> List[float]:
    """
    Parse a length-n vector from text.
    Accepts space/comma separated values (single line or multi-line).
    """
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != n:
        raise ValueError(f"Expected {n} values in the vector, got {len(parts)}.")
    return [float(x) for x in parts]


def row_sums(P: List[List[float]]) -> List[float]:
    return [sum(r) for r in P]


def normalize_rows(P: List[List[float]]) -> List[List[float]]:
    out = []
    for r in P:
        s = sum(r)
        if abs(s) < EPS:
            raise ValueError("Cannot normalize a row with sum 0.")
        out.append([v / s for v in r])
    return out


def normalize_vector(p: List[float]) -> List[float]:
    s = sum(p)
    if abs(s) < EPS:
        raise ValueError("Cannot normalize a vector with sum 0.")
    return [v / s for v in p]


def validate_markov(P: List[List[float]], p0: List[float]) -> Tuple[bool, str]:
    n = len(P)
    # Non-negativity
    if any(v < -1e-12 for row in P for v in row):
        return False, "Transition matrix has negative entries."
    if any(v < -1e-12 for v in p0):
        return False, "Initial vector has negative entries."

    # Row-stochastic
    sums = row_sums(P)
    bad_rows = [i for i, s in enumerate(sums) if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-6)]
    if bad_rows:
        return False, f"Some matrix rows do not sum to 1 (rows: {', '.join(str(i+1) for i in bad_rows)})."

    # Vector sums to 1
    if not math.isclose(sum(p0), 1.0, rel_tol=0.0, abs_tol=1e-6):
        return False, "Initial vector does not sum to 1."

    # Dimension check
    if any(len(r) != n for r in P) or len(p0) != n:
        return False, "Dimension mismatch."

    return True, "OK"


def markov_predict(P: List[List[float]], p0: List[float], steps: int = 5) -> List[List[float]]:
    """
    Uses row-vector convention: p_{t+1} = p_t * P
    Returns [p0, p1, ..., p_steps]
    """
    n = len(P)

    def step(p):
        return [sum(p[j] * P[j][i] for j in range(n)) for i in range(n)]

    states = [p0[:]]
    p = p0[:]
    for _ in range(steps):
        p = step(p)
        states.append(p)
    return states


def format_states(states: List[List[float]], labels: List[str] | None = None) -> str:
    n = len(states[0])
    if labels is None or len(labels) != n:
        labels = [f"S{i}" for i in range(n)]

    lines = []
    for t, p in enumerate(states):
        lines.append(f"Step {t}:")
        for name, val in zip(labels, p):
            lines.append(f"  {name}: {val:.6f}")
        lines.append(f"  (sum = {sum(p):.6f})")
        lines.append("")
    return "\n".join(lines)


class MarkovUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Markov Chain Predictor (5 steps)")
        self.geometry("900x650")

        self.n_var = tk.StringVar(value="3")
        self.normalize_var = tk.BooleanVar(value=False)
        self.labels_var = tk.StringVar(value="S0 S1 S2")

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Number of states (N):").pack(side="left")
        ttk.Entry(top, textvariable=self.n_var, width=6).pack(side="left", padx=(6, 16))

        ttk.Checkbutton(top, text="Auto-normalize (rows + vector)", variable=self.normalize_var).pack(side="left")

        ttk.Label(top, text="State labels (optional):").pack(side="left", padx=(16, 6))
        ttk.Entry(top, textvariable=self.labels_var, width=30).pack(side="left")

        middle = ttk.Frame(self, padding=10)
        middle.pack(fill="both", expand=True)

        # Matrix input
        left = ttk.Frame(middle)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ttk.Label(left, text="Transition Matrix P (N lines, N numbers each; rows sum to 1):").pack(anchor="w")
        self.matrix_text = tk.Text(left, height=14, wrap="none")
        self.matrix_text.pack(fill="both", expand=True)

        # Vector input
        right = ttk.Frame(middle)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(right, text="Initial Distribution p0 (N numbers; sums to 1):").pack(anchor="w")
        self.vector_text = tk.Text(right, height=6, wrap="none")
        self.vector_text.pack(fill="x", expand=False)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Predict 5 steps", command=self.on_predict).pack(side="left")
        ttk.Button(btns, text="Load example", command=self.load_example).pack(side="left", padx=8)
        ttk.Button(btns, text="Clear", command=self.clear_all).pack(side="left")

        ttk.Label(right, text="Output:").pack(anchor="w")
        self.output = tk.Text(right, height=18, wrap="none", state="disabled")
        self.output.pack(fill="both", expand=True)

        self.load_example()

    def clear_all(self):
        self.matrix_text.delete("1.0", "end")
        self.vector_text.delete("1.0", "end")
        self._set_output("")

    def load_example(self):
        # Example Markov chain with 3 states
        # Row-stochastic matrix
        example_P = (
            "0.70 0.20 0.10\n"
            "0.10 0.80 0.10\n"
            "0.20 0.30 0.50\n"
        )
        example_p0 = "1.0 0.0 0.0"

        self.matrix_text.delete("1.0", "end")
        self.matrix_text.insert("1.0", example_P)
        self.vector_text.delete("1.0", "end")
        self.vector_text.insert("1.0", example_p0)

        self.n_var.set("3")
        self.labels_var.set("Sunny Cloudy Rainy")
        self.normalize_var.set(False)
        self._set_output("Loaded an example Markov chain.\nClick 'Predict 5 steps'.")

    def _set_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def on_predict(self):
        try:
            n = int(self.n_var.get().strip())
            if n <= 0:
                raise ValueError("N must be a positive integer.")
        except Exception:
            messagebox.showerror("Invalid N", "Please enter a valid positive integer for N.")
            return

        P_text = self.matrix_text.get("1.0", "end").strip()
        p0_text = self.vector_text.get("1.0", "end").strip()

        try:
            P = parse_matrix(P_text, n)
            p0 = parse_vector(p0_text, n)

            if self.normalize_var.get():
                P = normalize_rows(P)
                p0 = normalize_vector(p0)

            ok, msg = validate_markov(P, p0)
            if not ok:
                raise ValueError(msg)

            labels = self.labels_var.get().split()
            if len(labels) != n:
                labels = [f"S{i}" for i in range(n)]

            states = markov_predict(P, p0, steps=5)
            out = format_states(states, labels=labels)

            self._set_output(out)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = MarkovUI()
    app.mainloop()
