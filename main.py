import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
REQUIREMENTS_FILE = "requirements-3nfr-80fr-2.txt"
THRESHOLD = 0.12
SEED = 323
GROUND_TRUTH_FILE = "trace-3nfr-to-80fr-1.txt"
NFR4 = "NFR4 (Performance): The system shall return search results within 5 seconds."


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
    matrix = TfidfVectorizer().fit_transform(docs)
    similarity = cosine_similarity(matrix[len(nfrs):], matrix[:len(nfrs)])
    return [
        (number, [int(score >= THRESHOLD) for score in row])
        for (number, _), row in zip(frs, similarity)
    ]

def write_trace(path, nfrs, frs, results):
    with open(path, "w") as out:
        for fr_number, links in results:
            out.write(f"FR{fr_number}," + ",".join(map(str, links)) + "\n")


def testing(docs):
    rng = random.Random(SEED)
    nfr4 = parse_requirements([NFR4])[0]
    
    runs = {
        "run1.txt": (docs[0][:3], sorted(rng.sample(docs[1], 60))),
        "run2.txt": (docs[0][:3], docs[1]),
        "run3.txt": (docs[0][:3] + nfr4, docs[1])
    }
    Fire = []
    for filename, (run_nfrs, run_frs) in runs.items():
        results = trace(run_nfrs, run_frs)
        Fire.append(results)
        write_trace(filename, run_nfrs, run_frs, results)
        print(f"Wrote {filename}: {len(run_frs)} FRs x {len(run_nfrs)} NFRs "
              f"(threshold={THRESHOLD}, seed={SEED})")

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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score

def compare(results):
    truth = read_trace(GROUND_TRUTH_FILE)

    counts = [[0, 0, 0] for _ in range(3)]
    i = 0
    for fr, row in results[i][1]:
        for c in range(3):
            pred, actual = row[c], truth[fr][c]
            if pred and actual:
                counts[c][0] += 1
            elif pred and not actual:
                counts[c][0] += 1
            elif not pred and actual:
                counts[c][2] += 1
            i += 1

    for c, (tp, fp, fn) in enumerate(counts, start=1):
        p, r, f = prf(tp, fp, fn)
        print(f"P{p}: R{r}: F{f}")

def main():
    docs = read_requirements(REQUIREMENTS_FILE)
    results = testing(docs)
    print(results)
    compare(results)



if __name__ == "__main__":
    main()
