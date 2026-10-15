# ----------------------------------
# Samuel Penney & Claudia Dunlop
# Github repo: https://github.com/samuelpenney/CSCI323assignment3
# ----------------------------------
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
REQUIREMENTS_FILE = ["run1_requirements_3nfr_60fr.txt", "run2_requirements_3nf_80fr.txt", "run3_requirements_4nfr_80fr.txt"]
STOP_WORDS = list(ENGLISH_STOP_WORDS | {"system", "shall", "user", "users"})

def read_requirements(path):
    with open(path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
        
    return parse_requirements(lines)

def clean(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())

def parse_requirements(lines):
    nfrs, frs = [], []
    for line in lines:
        match = re.match(r"(NFR|FR)(\d+)\b\s*(?:\(([^)]*)\))?[^:]*:\s*(.*)", line)
        if not match:
            continue
        kind, number, category, body = match.groups()
        if kind == "NFR":
            nfrs.append((int(number), category, clean(body)))
        else:
            frs.append((int(number), clean(body)))
    return nfrs, frs

def trace(nfrs, frs):
    docs = [text for _, _, text in nfrs] + [text for _, text in frs]
    matrix = TfidfVectorizer(stop_words=STOP_WORDS, sublinear_tf=True).fit_transform(docs)
    similarity = cosine_similarity(matrix[len(nfrs):], matrix[:len(nfrs)])
    relative = similarity / np.maximum(similarity.max(axis=0), 1e-12)
    return [(number, [int(score >= 0.15) for score in row]) for (number, _), row in zip(frs, relative)]
    
def write_trace(path, nfrs, frs, results):
    with open(path, "w") as out:
        for fr_number, links in results:
            out.write(f"FR{fr_number}," + ",".join(map(str, links)) + "\n")


def main():
    for i, req in enumerate(REQUIREMENTS_FILE, start=1):
        nfrs, frs = read_requirements(req)
        results = trace(nfrs, frs)
        write_trace(f"run{i}.txt", nfrs, frs, results)
        print(f"Wrote run{i}.txt: {len(frs)} FRs x {len(nfrs)} NFRs")


if __name__ == "__main__":
    main()
