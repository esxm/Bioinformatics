import json
import re
import string
from collections import Counter, defaultdict

# -----------------------------
# 1) Use a fixed English text (~300 characters, incl. spaces + punctuation)
#    (exactly 300 characters)
# -----------------------------
TEXT_300 = (
    "On Friday evening, the streetlights flickered; a small crowd waited quietly. "
    "Someone laughed, someone sighed, and a dog shook off rain. "
    "We'll be fine, she said—then the bus arrived late, as usual, but warm. "
    "Inside, windows fogged and coats steamed; the driver hummed an old tune. "
    "A child counted stops on sticky fingers while the city slid by, blurred and bright."
)


# -----------------------------
# 2) Tokenize into "words"
#    (If you want ONLY word-to-word transitions, keep punctuation out.)
# -----------------------------
def tokenize_words_only(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

# -----------------------------
# 3) Map each unique word to an ASCII symbol
# -----------------------------
def make_ascii_symbol_map(tokens):
    ascii_symbols = [chr(i) for i in range(33, 127)]  # printable ASCII, no whitespace
    uniq = list(dict.fromkeys(tokens))  # preserve order
    if len(uniq) > len(ascii_symbols):
        raise ValueError(f"Too many unique tokens ({len(uniq)}). Increase symbol pool.")
    token_to_sym = {tok: ascii_symbols[i] for i, tok in enumerate(uniq)}
    sym_to_token = {v: k for k, v in token_to_sym.items()}
    return token_to_sym, sym_to_token

# -----------------------------
# 4) Build transition probabilities P(next | current)
# -----------------------------
def transition_probabilities(tokens, token_to_sym):
    seq = [token_to_sym[t] for t in tokens]
    counts = defaultdict(Counter)
    for cur, nxt in zip(seq, seq[1:]):
        counts[cur][nxt] += 1

    probs = {}
    for cur, ctr in counts.items():
        total = sum(ctr.values())
        probs[cur] = {nxt: c / total for nxt, c in ctr.items()}
    return probs

# -----------------------------
# 5) Save JSON
# -----------------------------
def main():
    print("Fixed text (exactly 300 chars):")
    print(TEXT_300)
    print("\nCharacter count:", len(TEXT_300))

    tokens = tokenize_words_only(TEXT_300)
    token_to_sym, sym_to_token = make_ascii_symbol_map(tokens)
    matrix = transition_probabilities(tokens, token_to_sym)

    payload = {
        "text": TEXT_300,
        "tokenization_note": "Words only (punctuation excluded); all lowercased.",
        "symbol_to_token": sym_to_token,
        "transition_matrix": matrix
    }

    out_file = "transition_matrix.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nSaved transition matrix to:", out_file)
    print("Unique words:", len(sym_to_token))

    # Show sample transitions
    if matrix:
        some_state = next(iter(matrix))
        print(f"\nSample transitions from symbol '{some_state}' ({sym_to_token[some_state]!r}):")
        for k, v in sorted(matrix[some_state].items(), key=lambda x: -x[1])[:10]:
            print(f"  -> '{k}' ({sym_to_token[k]!r}): {v:.3f}")

if __name__ == "__main__":
    main()