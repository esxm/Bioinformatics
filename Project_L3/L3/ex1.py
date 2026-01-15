import math

def computeFreq(seq: str, c: str) -> float:
    """Frequency of base c in seq (0..1)."""
    c = c.upper()              # <-- call the method
    seq = seq.upper()          # <-- assign the uppercased string
    if not seq:
        return 0.0
    return seq.count(c) / len(seq)

def computeCount(seq: str, c: str) -> int:
    """Count of base c in seq."""
    return seq.upper().count(c.upper())

def computeMeltingTempSimple(seq: str) -> float:
    # Wallace rule uses COUNTS, not frequencies
    A = seq.count("A")
    C = seq.count("C")
    G = seq.count("G")
    T = seq.count("T")
    return 4 * (C + G) + 2 * (A + T)

def computeMeltingTempComplex(seq: str, na_molar: float = 0.001) -> float:
   
    s = seq.upper()
    if na_molar <= 0:
        raise ValueError("[Na+] must be positive (mol/L).")
    length = len(s)
    if length == 0:
        raise ValueError("Sequence is empty.")
    # %GC (0..100)
    gc_pct = 100.0 * (s.count("G") + s.count("C")) / length
    return 81.5 + 16.6 * math.log10(na_molar) + 0.41 * gc_pct - (600.0 / length)

seq = "ATTTCGCCGATA"
print(f"Computing melting temperature with simple formula: {computeMeltingTempSimple(seq)}C°")
print(f"Computing melting temperature with complex formula: {computeMeltingTempComplex(seq):.2f}C°")
