import csv
import math
from collections import Counter
from pathlib import Path


# --------------------------------------------------------------------------
# STEP 0 — Data Loading
# --------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """Load job roles and their associated skill tags from a CSV file.

    Expected CSV columns: role, skills (space-separated tags).
    Returns a list of dicts: {"role": str, "tokens": list[str]}
    """
    dataset = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = _tokenize(row["skills"])
            dataset.append({"role": row["role"], "tokens": tokens})
    return dataset


def _tokenize(text: str) -> list[str]:
    """Normalize a raw skills string into a clean list of lowercase tokens."""
    return [t.strip().lower() for t in text.split() if t.strip()]


# --------------------------------------------------------------------------
# STEP 1 — Vector Mapping (bridging the language barrier)
# --------------------------------------------------------------------------

def build_vocabulary(dataset: list[dict]) -> list[str]:
    """Build the shared vocabulary space every profile (user + item) maps to."""
    vocab = set()
    for item in dataset:
        vocab.update(item["tokens"])
    return sorted(vocab)


# --------------------------------------------------------------------------
# STEP 2 — TF-IDF Weighting (upgrading past binary 1s and 0s)
# --------------------------------------------------------------------------

def compute_idf(dataset: list[dict], vocab: list[str]) -> dict:
    """IDF(t) = log(Total Documents / Documents containing term t)

    A small +1 smoothing is applied to the denominator so unseen terms
    never trigger a divide-by-zero (a common cold-start edge case).
    """
    n_docs = len(dataset)
    doc_freq = Counter()
    for term in vocab:
        for item in dataset:
            if term in item["tokens"]:
                doc_freq[term] += 1

    idf = {}
    for term in vocab:
        idf[term] = math.log(n_docs / (1 + doc_freq[term])) + 1  # smoothed IDF
    return idf


def compute_tf(tokens: list[str], vocab: list[str]) -> dict:
    """TF(t) = (count of term t in document) / (total terms in document)"""
    total = len(tokens) or 1  # guard against empty (cold-start) profiles
    counts = Counter(tokens)
    return {term: counts.get(term, 0) / total for term in vocab}


def vectorize(tokens: list[str], vocab: list[str], idf: dict) -> list[float]:
    """Combine TF and IDF into the final weighted vector for a profile."""
    tf = compute_tf(tokens, vocab)
    return [tf[term] * idf[term] for term in vocab]


# --------------------------------------------------------------------------
# STEP 3 — Similarity Engine (Cosine Similarity, the industry standard)
# --------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """cos(theta) = (A . B) / (||A|| * ||B||)

    Invariant to vector magnitude -> measures orientation/alignment of
    interests rather than raw overlap count, fixing the Euclidean-distance
    scale problem described in the training deck.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Cold Start: a zero vector can share no orientation
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------
# STEP 4 — The 4-Step Ranking Pipeline (Ingestion -> Scoring -> Sorting -> Filtering)
# --------------------------------------------------------------------------

class TechStackRecommender:
    """Content-based recommendation engine mapping skills -> job roles."""

    def __init__(self, dataset_path: str):
        self.dataset = load_dataset(dataset_path)
        self.vocab = build_vocabulary(self.dataset)
        self.idf = compute_idf(self.dataset, self.vocab)

        # Pre-compute item vectors once (they don't change per query)
        for item in self.dataset:
            item["vector"] = vectorize(item["tokens"], self.vocab, self.idf)

    def recommend(self, user_skills: list[str], top_n: int = 3) -> dict:
        """Run the full IPO pipeline and return the Top-N ranked job roles.

        Args:
            user_skills: raw skill strings from the user (min. 3 recommended)
            top_n: how many results to return (default 3, per the capstone)
        """
        # --- Step 1: Ingestion ---
        user_tokens = _tokenize(" ".join(user_skills))
        recognized = [t for t in user_tokens if t in self.vocab]
        unrecognized = [t for t in user_tokens if t not in self.vocab]

        user_vector = vectorize(user_tokens, self.vocab, self.idf)

        # --- Step 2: Scoring ---
        scored = []
        for item in self.dataset:
            score = cosine_similarity(user_vector, item["vector"])
            scored.append({"role": item["role"], "score": score})

        # --- Step 3: Sorting ---
        scored.sort(key=lambda x: x["score"], reverse=True)

        # --- Step 4: Filtering ---
        top_results = scored[:top_n]

        return {
            "recognized_skills": recognized,
            "unrecognized_skills": unrecognized,
            "recommendations": top_results,
        }


# --------------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------------

def _print_report(result: dict) -> None:
    print("\n" + "=" * 50)
    print(" TECH STACK RECOMMENDER — RESULTS")
    print("=" * 50)

    if result["unrecognized_skills"]:
        print(f"⚠  Skills not in vocabulary (ignored): {', '.join(result['unrecognized_skills'])}")

    print(f"✔  Matched on: {', '.join(result['recognized_skills']) or 'none'}\n")

    print("Top Career Matches:")
    for i, rec in enumerate(result["recommendations"], start=1):
        pct = rec["score"] * 100
        bar = "█" * int(pct // 5)
        print(f"  {i}. {rec['role']:<28} {pct:5.1f}%  {bar}")
    print()


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Recommend career paths based on your skills using TF-IDF + Cosine Similarity."
    )
    parser.add_argument(
        "skills", nargs="*",
        help='Your skills, e.g. python "cloud computing" automation'
    )
    parser.add_argument(
        "--data", default="/content/raw_skills.csv", # Use the existing raw_skills.csv file
        help="Path to the job-roles dataset CSV (default: raw_skills.csv)"
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Number of recommendations to return (default: 3)"
    )

    # In a Colab/IPython environment, the kernel often passes arguments like '-f'
    # which argparse might not recognize. We parse only known arguments.
    args, unknown = parser.parse_known_args()

    engine = TechStackRecommender(args.data)

    skills_to_use = args.skills # Start with whatever argparse found

    # If no skills were provided via command line, or too few, use a default set.
    if not skills_to_use or len(skills_to_use) < 3:
        print("No or insufficient command-line skills provided. Using default skills for demonstration.")
        skills_to_use = ["python", "machine learning", "sql"]

    result = engine.recommend(skills_to_use, top_n=args.top)
    _print_report(result)


if __name__ == "__main__":
    main()