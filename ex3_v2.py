#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict

# Full IUPAC DNA set + gap, tracked individually (order for reporting)
IUPAC_ORDER = tuple(b"A C G T U R Y K M S W B D H V N -".split())
IUPAC_SET = set(b"ACGTUR YKMSWBDHVN-".replace(b" ", b""))  # bytes set
ALL_KEYS = tuple([k.decode() for k in IUPAC_ORDER] + ["Invalid"])

def pct(x, total):
    return (x / total * 100.0) if total else 0.0

class SeqStats:
    def __init__(self, header):
        self.header = header
        self.total = 0
        # per-symbol counts (A,C,G,T,U,R,Y,K,M,S,W,B,D,H,V,N,'-') + Invalid
        self.counts = {k.decode(): 0 for k in IUPAC_ORDER}
        self.counts["Invalid"] = 0
        self.lines_ok_80 = True
        self._last_line_len = 0
        self._saw_len_violation = False

    def add_line(self, line_bytes):
        # normalize to uppercase to count everything
        line_bytes = line_bytes.upper()
        ln = len(line_bytes)
        if ln != 80:
            self._saw_len_violation = True
        self._last_line_len = ln
        self.total += ln

        # Fast pre-count for the most common letters
        c = self.counts
        c["A"] += line_bytes.count(b"A")
        c["C"] += line_bytes.count(b"C")
        c["G"] += line_bytes.count(b"G")
        c["T"] += line_bytes.count(b"T")
        c["U"] += line_bytes.count(b"U")
        c["N"] += line_bytes.count(b"N")

        # Count everything else precisely (ambiguity codes, gaps, invalid)
        known = c["A"] + c["C"] + c["G"] + c["T"] + c["U"] + c["N"]
        if known < ln:
            for ch in line_bytes:
                if ch in (65, 67, 71, 84, 85, 78):  # A,C,G,T,U,N
                    continue
                if ch in IUPAC_SET:
                    c[bytes([ch]).decode()] += 1
                else:
                    c["Invalid"] += 1

    def finalize(self):
        # allow last line <80 chars per FASTA convention
        if self._last_line_len < 80:
            self.lines_ok_80 = True
        else:
            self.lines_ok_80 = not self._saw_len_violation

    def format_block(self):
        lines = []
        lines.append(f"Header: {self.header}")
        lines.append(f"Length (bases): {self.total:,}")
        gc = self.counts["G"] + self.counts["C"]
        lines.append(f"GC%: {pct(gc, self.total):.3f}%")
        lines.append(f"All internal lines 80 chars: {'Yes' if self.lines_ok_80 else 'No'}")
        lines.append("Composition (count | %):")
        for k in ALL_KEYS:
            lines.append(f"  {k:7s}: {self.counts[k]:>12,} | {pct(self.counts[k], self.total):6.3f}%")
        return "\n".join(lines)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FASTA Uploader (Multi-FASTA, full IUPAC counts)")
        self.geometry("880x600")

        self.btn = ttk.Button(self, text="Upload FASTA…", command=self.choose_file)
        self.btn.pack(pady=10)

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=760)
        self.progress.pack(pady=6)

        self.status = tk.StringVar(value="No file loaded.")
        ttk.Label(self, textvariable=self.status).pack(pady=2)

        self.output = tk.Text(self, wrap="word", height=26)
        self.output.pack(expand=True, fill="both", padx=10, pady=10)
        self.output.configure(state="disabled")

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select FASTA file",
            filetypes=[("FASTA", "*.fa *.fasta *.fna *.ffn *.frn *.faa"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            size = os.path.getsize(path)
            self.status.set(f"Selected: {os.path.basename(path)}  ({size/1024/1024:.2f} MiB)")
            self.progress.configure(value=0, maximum=max(1, size))
            self.output_to_user("Processing… large files may take a moment.\n")
            t = threading.Thread(target=self._process_file_thread, args=(path,), daemon=True)
            t.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _process_file_thread(self, path):
        try:
            def on_progress(done, total):
                self.after(0, self._update_progress, done, total)

            records, overall = self.parse_multi_fasta(path, on_progress)
            report = self.render_report(records, overall)
            self.after(0, self._show_results, report)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Processing error", str(e)))

    def _update_progress(self, done, total):
        self.progress["maximum"] = max(1, total)
        self.progress["value"] = min(done, total)
        self.status.set(f"Reading… {done/1024/1024:.2f} / {total/1024/1024:.2f} MiB")

    def _show_results(self, text):
        self.status.set("Done.")
        self.output_to_user(text)

    def output_to_user(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")

    # -------- parsing ----------
    def parse_multi_fasta(self, path, progress_cb=None):
        filesize = os.path.getsize(path)
        read_bytes = 0
        records = []
        current = None

        with open(path, "rb", buffering=1024*1024) as fh:
            first = fh.readline()
            read_bytes += len(first)
            if not first:
                raise ValueError("Empty file.")
            if not first.startswith(b">"):
                raise ValueError("Not a FASTA: first line must start with '>'")
            current = SeqStats(first[1:].decode("utf-8", "replace").strip())

            for line in fh:
                read_bytes += len(line)
                if progress_cb:
                    progress_cb(read_bytes, filesize)

                if line.startswith(b">"):
                    current.finalize()
                    records.append(current)
                    current = SeqStats(line[1:].decode("utf-8", "replace").strip())
                    continue

                line = line.rstrip(b"\r\n")
                if line:
                    current.add_line(line)

            # EOF
            if current is not None:
                current.finalize()
                records.append(current)

        # overall
        overall_counts = defaultdict(int)
        overall_total = 0
        for r in records:
            overall_total += r.total
            for k, v in r.counts.items():
                overall_counts[k] += v
        return records, (overall_total, dict(overall_counts))

    def render_report(self, records, overall):
        out = []
        for i, r in enumerate(records, 1):
            out.append("-" * 76)
            out.append(f"Record {i}")
            out.append(r.format_block())
        out.append("-" * 76)
        total, oc = overall
        out.append("OVERALL SUMMARY")
        out.append(f"Total length (all sequences): {total:,}")
        gc = oc.get("G", 0) + oc.get("C", 0)
        out.append(f"Overall GC%: {pct(gc, total):.3f}%")
        out.append("Overall composition (count | %):")
        for k in ALL_KEYS:
            out.append(f"  {k:7s}: {oc.get(k,0):>12,} | {pct(oc.get(k,0), total):6.3f}%")
        return "\n".join(out)

if __name__ == "__main__":
    App().mainloop()
