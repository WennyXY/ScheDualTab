from typing import List

def get_prompt_dict(prompt_file):
    import yaml
    
    with open(prompt_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def aggregate_sc_by_voting(results: List[str]) -> str:
    if len(results) == 0:
        return ''
    counts = {}
    for result in results:
        if result in counts:
            counts[result] += 1
        else:
            counts[result] = 1
    return max(counts, key=counts.get)
    
def aggregate_sc_by_fuzzy_match(results:List[str]) -> str:
    # fuzzy matching
    if len(results) == 0:
        return ''
    from rapidfuzz import fuzz, process
    counts = {}
    for result in results:
        flag = 0
        for key in counts.keys():
            if fuzz.ratio(result, key) > 90:
                counts[key] += 1
                flag = 1
                break
        if flag == 0:
            counts[result] = 1
    return max(counts, key=counts.get)


def row_to_sentence(row):
    sentence = ", ".join([f"{col} is {row[col]}" for col in row.index]) + "."
    
    return sentence

