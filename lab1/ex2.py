"""
Design an application which finds the relative frequencies for the symbols 
found in the alphabet of sequence S

"""
sequence = "ATTTCGCCGATA"

def find_alphabet(string: str):
    seen = {}  # dictionary to store unique characters
    for char in string:
        if char not in seen:
            seen[char] = True   # mark as seen
    return list(seen.keys())   # extract unique characters

alphabet = find_alphabet(sequence)

def relative_freq(alphabet: list, string: str):
    result = {}
    for c in alphabet:
        result[c] = str(100 * len([n for n in string if(n == c)])/len(string)) + "%"

    return result

print(relative_freq(alphabet=alphabet, string=sequence))