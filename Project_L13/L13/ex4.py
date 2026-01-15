import json
import random
import re

def load_transition_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = data["transition_matrix"]          # dict: cur_symbol -> {next_symbol: prob}
    sym_to_token = data["symbol_to_token"]      # dict: symbol -> word
    return matrix, sym_to_token

def weighted_choice(rng: random.Random, next_probs: dict) -> str:
    """next_probs: {symbol: prob} where probs sum to ~1."""
    r = rng.random()
    cum = 0.0
    last = None
    for sym, p in next_probs.items():
        cum += p
        last = sym
        if r <= cum:
            return sym
    return last  # fallback for tiny floating-point drift

def synthesize_symbols(matrix: dict, length: int, rng: random.Random, start_symbol: str | None = None):
    states = list(matrix.keys())
    if not states:
        return []

    # pick a start state (or validate provided one)
    cur = start_symbol if (start_symbol in matrix) else rng.choice(states)
    seq = [cur]

    for _ in range(length - 1):
        next_probs = matrix.get(cur)
        if not next_probs:
            # dead-end: restart from a random state that has outgoing transitions
            cur = rng.choice(states)
            seq.append(cur)
            continue
        cur = weighted_choice(rng, next_probs)
        seq.append(cur)

    return seq

def symbols_to_text(symbol_seq: list[str], sym_to_token: dict):
    words = [sym_to_token.get(s, "") for s in symbol_seq]
    words = [w for w in words if w]

    # If your original model was "words only", just join with spaces.
    text = " ".join(words)

    # Light cleanup: collapse spaces before punctuation if punctuation ever appears as tokens.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text

def main():
    json_path = "transition_matrix.json"
    out_words = 80
    seed = 123

    matrix, sym_to_token = load_transition_model(json_path)
    rng = random.Random(seed)

    sym_seq = synthesize_symbols(matrix, length=out_words, rng=rng)
    new_text = symbols_to_text(sym_seq, sym_to_token)

    print("\n\nThe newly generated text: \n")
    print(new_text + "\n\n")

if __name__ == "__main__":
    main()