#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

VALID_CHARS = set(b"ACGTURYKMSWBDHVN-")  # IUPAC DNA + gap, uppercase only

class FastaStats:
    def __init__(self):
        self.header = ""
        self.total_bases = 0
        self.counts = {
            "A": 0, "C": 0, "G": 0, "T": 0,
            "N": 0, "Other": 0
        }
        self.lines_ok_80 = True
        self.last_line_len = 0

    def as_text(self):
        lines = [f"Header: {self.header}",
                 f"Total length (bases): {self.total_bases:,}",
                 f"All internal lines 80 chars: {'Yes' if self.lines_ok_80 else 'No'}",
                 f"Final line length: {self.last_line_len} (last line may be <80)"]

        # Add per-character counts & percentages
        lines.append("\nBase composition:")
        for base, count in self.counts.items():
            if self.total_bases > 0:
                perc = count / self.total_bases * 100
            else:
                perc = 0.0
            lines.append(f"  {base}: {count:,} ({perc:.3f}%)")
        return "\n".join(lines)


def process_fasta_stream(path, progress_cb=None):
    """Stream through a potentially huge FASTA without loading it into memory."""
    stats = FastaStats()
    file_size = os.path.getsize(path)
    bytes_read = 0

    with open(path, "rb", buffering=1024 * 1024) as fh:
        # Header
        header = fh.readline()
        bytes_read += len(header)
        if not header.startswith(b">"):
            raise ValueError("Not a FASTA file: first line must start with '>'")
        stats.header = header[1:].decode("utf-8", "replace").strip()

        # Sequence
        while True:
            line = fh.readline()
            if not line:
                break
            bytes_read += len(line)
            if progress_cb:
                progress_cb(bytes_read, file_size)

            line = line.rstrip(b"\r\n")
            if not line:
                continue

            if len(line) != 80:
                stats.lines_ok_80 = False
            stats.last_line_len = len(line)

            stats.total_bases += len(line)

            # Count bases
            for ch in line:
                if ch == 65:   # 'A'
                    stats.counts["A"] += 1
                elif ch == 67: # 'C'
                    stats.counts["C"] += 1
                elif ch == 71: # 'G'
                    stats.counts["G"] += 1
                elif ch == 84: # 'T'
                    stats.counts["T"] += 1
                elif ch == 78: # 'N'
                    stats.counts["N"] += 1
                elif ch in VALID_CHARS:
                    # Other valid IUPAC ambiguity codes
                    stats.counts["Other"] += 1
                else:
                    stats.counts["Other"] += 1

        if stats.last_line_len < 80:
            stats.lines_ok_80 = True

    return stats


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FASTA Uploader (Huge-file friendly)")
        self.geometry("720x480")

        self.btn = ttk.Button(self, text="Upload FASTA…", command=self.choose_file)
        self.btn.pack(pady=12)

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=600)
        self.progress.pack(pady=6)

        self.status = tk.StringVar(value="No file loaded.")
        ttk.Label(self, textvariable=self.status).pack(pady=2)

        self.output = tk.Text(self, wrap="word", height=18)
        self.output.pack(expand=True, fill="both", padx=10, pady=10)
        self.output.configure(state="disabled")

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select FASTA file",
            filetypes=[("FASTA", "*.fa *.fasta *.fna"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            filesize = os.path.getsize(path)
            self.status.set(f"Selected: {os.path.basename(path)}  ({filesize/1024/1024:.2f} MiB)")
            self.progress.configure(value=0, maximum=filesize if filesize > 0 else 1)
            self.output_to_user("Processing… this may take a bit for very large files.\n")
            t = threading.Thread(target=self._process_file_thread, args=(path,), daemon=True)
            t.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _process_file_thread(self, path):
        try:
            def on_progress(bytes_read, total):
                self.after(0, self._update_progress, bytes_read, total)

            stats = process_fasta_stream(path, progress_cb=on_progress)
            text = stats.as_text()
            self.after(0, self._show_results, text)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Processing error", str(e)))

    def _update_progress(self, bytes_read, total):
        self.progress["maximum"] = max(1, total)
        self.progress["value"] = min(bytes_read, total)
        self.status.set(f"Reading… {bytes_read/1024/1024:.2f} / {total/1024/1024:.2f} MiB")

    def _show_results(self, text):
        self.status.set("Done.")
        self.output_to_user(text)

    def output_to_user(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
