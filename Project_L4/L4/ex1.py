# mRNA → protein translator (standard genetic code, works on U-based RNA)
# - Starts at first AUG
# - Stops at first in-frame UAA/UAG/UGA
# - Prints both 1-letter and 3-letter AA sequences

RNA_TO_AA_1 = {
    # U-row
    "UUU":"F","UUC":"F","UUA":"L","UUG":"L",
    "UCU":"S","UCC":"S","UCA":"S","UCG":"S",
    "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*",
    "UGU":"C","UGC":"C","UGA":"*","UGG":"W",
    # C-row
    "CUU":"L","CUC":"L","CUA":"L","CUG":"L",
    "CCU":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAU":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "CGU":"R","CGC":"R","CGA":"R","CGG":"R",
    # A-row
    "AUU":"I","AUC":"I","AUA":"I","AUG":"M",
    "ACU":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAU":"N","AAC":"N","AAA":"K","AAG":"K",
    "AGU":"S","AGC":"S","AGA":"R","AGG":"R",
    # G-row
    "GUU":"V","GUC":"V","GUA":"V","GUG":"V",
    "GCU":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAU":"D","GAC":"D","GAA":"E","GAG":"E",
    "GGU":"G","GGC":"G","GGA":"G","GGG":"G",
}

AA1_TO_AA3 = {
    "A":"Ala","R":"Arg","N":"Asn","D":"Asp","C":"Cys","Q":"Gln","E":"Glu","G":"Gly",
    "H":"His","I":"Ile","L":"Leu","K":"Lys","M":"Met","F":"Phe","P":"Pro","S":"Ser",
    "T":"Thr","W":"Trp","Y":"Tyr","V":"Val","*":"Stop"
}

def translate_cds_mrna(rna: str) -> str:
    """Translate an mRNA (5'→3', U-based) from first AUG to first stop."""
    rna = "".join(rna.upper().replace("T","U").split())  # normalize
    start = rna.find("AUG")
    if start == -1:
        raise ValueError("No start codon (AUG) found.")
    aas = []
    for i in range(start, len(rna) - 2, 3):
        codon = rna[i:i+3]
        aa = RNA_TO_AA_1.get(codon)
        if aa is None:
            raise ValueError(f"Invalid/unknown codon: {codon} at pos {i+1}-{i+3}")
        if aa == "*":
            break
        aas.append(aa)
    return "".join(aas)

def aa1_to_aa3(aa_seq: str) -> str:
    return "-".join(AA1_TO_AA3[a] for a in aa_seq)

# ----- Hardcoded mRNA with exactly one AUG and one in-frame stop -----
mrna_seq = "GGCUUCCGCGGCUUUGGCC AUG GCU GAA CCU UUC CCC GGC ACU UAA CCGGGCUUCGGCCG"

aa_1 = translate_cds_mrna(mrna_seq)
aa_3 = aa1_to_aa3(aa_1)

print("mRNA (5'→3'):", mrna_seq)
print("AA (1-letter):", aa_1)      
print("AA (3-letter):", aa_3)      
