
sequence = "ATTTCGCCGATA"

def find_alphabet(string: str):
    seen = {}  # dictionary to store unique characters
    for char in string:
        if char not in seen:
            seen[char] = True   # mark as seen
    return list(seen.keys())   # extract unique characters

print(find_alphabet(sequence))