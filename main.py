# ----------------------------------
# Samuel Penney & Claudia Dunlop
# Github repo: https://github.com/samuelpenney/CSCI323assignment3
# ----------------------------------
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
REQUIREMENTS_FILE = "requirements-3nfr-80fr-2.txt"
THRESHOLDS = {1: 0.108, 2: 0.008, 3: 0.184, 4: 0.10}
SEED = 323
GROUND_TRUTH_FILE = "trace-3nfr-to-80fr-1.txt"
NFR4 = "NFR4 (Performance): The system shall return search results within 5 seconds."
STOP_WORDS = list(ENGLISH_STOP_WORDS | {"system", "shall", "user", "users"})

def read_requirements(path):
    with open(path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
        
    return parse_requirements(lines)

def parse_requirements(lines):
    nfrs, frs = [], []
    for line in lines:
        match = re.match(r"(NFR|FR)(\d+)\b[^:]*:\s*(.*)", line)
        if not match:
            continue
        kind, number, body = match.groups()
        entry = (int(number), re.sub(r"[^a-z0-9\s]", " ", body.lower()))
        (nfrs if kind == "NFR" else frs).append(entry)
    return nfrs, frs


def trace(nfrs, frs):
    docs = [text for _, text in nfrs] + [text for _, text in frs]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
    similarity = cosine_similarity(matrix[len(nfrs):], matrix[:len(nfrs)])
    return [(number, [int(score >= THRESHOLDS[nfr_num]) for (nfr_num, _), score in zip(nfrs, row)]) for (number, _), row in zip(frs, similarity)]

def write_trace(path, nfrs, frs, results):
    with open(path, "w") as out:
        for fr_number, links in results:
            out.write(f"FR{fr_number}," + ",".join(map(str, links)) + "\n")


def testing(docs):
    rng = random.Random(SEED)
    nfr4 = parse_requirements([NFR4])[0]

    runs = {"run1.txt": (docs[0][:3], sorted(rng.sample(docs[1], 60))), "run2.txt": (docs[0][:3], docs[1]), "run3.txt": (docs[0][:3] + nfr4, docs[1])}
    Fire = []
    for filename, (run_nfrs, run_frs) in runs.items():
        results = trace(run_nfrs, run_frs)
        Fire.append(results)
        write_trace(filename, run_nfrs, run_frs, results)
        print(f"Wrote {filename}: {len(run_frs)} FRs x {len(run_nfrs)} NFRs ")
        
        
    return Fire
    

def read_trace(path):
    links = {}
    
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            links[int(parts[0][2:])] = [int(x) for x in parts[1:]]
            
    return links

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1_score

def compare(results, runs):
    truth = read_trace(GROUND_TRUTH_FILE)

    counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for fr, row in results[runs]:
        for c, actual in enumerate(truth[fr]):
            pred = row[c]
            if pred and actual:
                counts[c][0] += 1
            elif pred and not actual:
                counts[c][1] += 1
            elif not pred and actual:
                counts[c][2] += 1

    print(f"\nRun {runs + 1}")
    
    for c, (tp, fp, fn) in enumerate(counts, start=1):
        p, r, f = prf(tp, fp, fn)
        print(f"NFR{c}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
    tp, fp, fn = (sum(count[c] for count in counts) for c in range(3))
    p, r, f = prf(tp, fp, fn)
    print(f"Overall: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

def main():
    docs = read_requirements(REQUIREMENTS_FILE)
    results = testing(docs)
    for i in range(3):
        compare(results, i)



if __name__ == "__main__":
    main()
