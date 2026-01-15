S = "ATTGTCCCAATCTGTTG"

def calculate_frequency(seq: str, k: int):
    index = 0
    total_k = 0
    table = {}

    while(index+k <= len(seq)):
        total_k += 1

        temp = seq[index:index+k]
        if(temp in table):
            table[temp] += 1
        else:
            table[temp] = 1

        index += 1

    for key in table:
        table[key] = f"{100 * table[key] / total_k:.4}%" 
        

    return dict(sorted(table.items()))  


print(calculate_frequency(S,2))
print(calculate_frequency(S,3))
