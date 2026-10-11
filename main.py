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



def main():
    pass

if __name__ == "__main__":
    main()
