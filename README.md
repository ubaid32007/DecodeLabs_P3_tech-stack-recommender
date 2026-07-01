# Tech Stack Recommender

**DecodeLabs Industrial Training — Artificial Intelligence Track, Project 3**

A content-based filtering recommendation engine that maps a user's raw skills
and career interests to the most relevant tech job roles, built entirely from
first principles using **TF-IDF weighting** and **Cosine Similarity** — no
`scikit-learn` required.

## How It Works

The engine follows the 4-step **IPO (Input → Process → Output) pipeline**:

| Step | Stage | What Happens |
|------|-------|---------------|
| 1 | **Ingestion** | Capture at least 3 raw user skills |
| 2 | **Scoring** | Convert skills → TF-IDF vectors, score every job role via Cosine Similarity |
| 3 | **Sorting** | Rank all job roles by descending similarity score |
| 4 | **Filtering** | Truncate to the Top-N most relevant matches |

### Why Content-Based Filtering?

Unlike collaborative filtering (which needs historical data from many users),
content-based filtering maps user input directly to item *attributes*. That
means it works immediately — even for a brand-new user — and it's naturally
resistant to the **item cold-start problem**, since new job roles just need
skill tags, not interaction history.

### Why TF-IDF over simple binary matching?

Plain "1 if present, 0 if absent" overlap scoring treats a generic skill like
`python` the same as a highly specific one like `kubernetes`. TF-IDF fixes
this by:
- **TF** — rewarding skills that appear more within a role's profile
- **IDF** — penalizing skills so common across roles that they carry little
  discriminating power

### Why Cosine Similarity over Euclidean distance?

Euclidean distance is sensitive to *magnitude* (a role with more listed
skills looks "farther away" even if perfectly aligned). Cosine similarity
instead measures the *angle* between vectors, so it captures orientation
(relevance) independent of how many total skills a role lists.

## Project Structure

```
tech-stack-recommender/
├── recommender.py     # Core engine: TF-IDF, cosine similarity, IPO pipeline, CLI
├── raw_skills.csv      # Dataset — 20 job roles mapped to skill tags
├── requirements.txt    # Dependencies (stdlib only — zero external deps)
└── README.md
```

## Usage

> **Note:** By default the script looks for the dataset at
> `/content/raw_skills.csv` (Google Colab's default upload path). If you're
> running locally, either place `raw_skills.csv` at that path or pass
> `--data` with your own path (see below).

### Google Colab

1. Upload `recommender.py` and `raw_skills.csv` to your Colab session (they'll
   land in `/content/` by default).
2. Run:
   ```python
   !python recommender.py python "cloud computing" automation
   ```

### Command line (direct skills)

```bash
python3 recommender.py Python "Cloud Computing" Automation --data raw_skills.csv
```

```
==================================================
 TECH STACK RECOMMENDER — RESULTS
==================================================
✔  Matched on: python, cloud, computing, automation

Top Career Matches:
  1. Machine Learning Engineer     49.3%  █████████
  2. AI Research Engineer          49.3%  █████████
  3. Data Scientist                45.3%  █████████
```

### No skills provided

If you run the script with no arguments, or fewer than 3 skills, it falls
back to a default demo query (`python`, `machine learning`, `sql`) so the
pipeline always produces a runnable result — useful when running as a Colab
cell with no CLI arguments available:

```bash
python3 recommender.py --data raw_skills.csv
```

```
No or insufficient command-line skills provided. Using default skills for demonstration.
...
Top Career Matches:
  1. Data Scientist
  2. Machine Learning Engineer
  3. AI Research Engineer
```

### Custom dataset / result count

```bash
python3 recommender.py Python SQL Statistics --data my_roles.csv --top 5
```

### As a library

```python
from recommender import TechStackRecommender

engine = TechStackRecommender("raw_skills.csv")
result = engine.recommend(["Python", "Machine Learning", "TensorFlow"], top_n=3)

for rec in result["recommendations"]:
    print(rec["role"], rec["score"])
```

## Handling the Cold-Start Problem

- **Unrecognized skills** (not in the dataset's vocabulary) are safely
  ignored and reported back to the user rather than crashing the pipeline.
- **Zero-vector guard**: if a user's profile shares no vocabulary with the
  dataset, cosine similarity correctly returns `0.0` instead of dividing by
  zero.
- **Missing/insufficient input**: if fewer than 3 skills are supplied (or
  none at all), the script falls back to a default skill set so the
  pipeline always completes rather than failing — this also makes it robust
  against extra kernel-injected arguments (e.g. `-f`) in notebook
  environments like Colab/Jupyter, which are silently ignored via
  `parse_known_args()`.
- To bootstrap genuinely new users in production, this same engine could be
  paired with an onboarding survey or a "trending roles" fallback list, as
  discussed in the training material.

## Extending the Dataset

`raw_skills.csv` uses two columns:

```csv
role,skills
Data Scientist,"Python SQL Machine Learning Statistics Pandas NumPy"
```

Add new rows to expand the pool of recommendable job roles — no code changes
needed, since the vocabulary and IDF weights are rebuilt automatically from
whatever CSV is loaded.

## Author

Ubaid — CS Undergraduate, UET Mardan | DecodeLabs Industrial Training (2026 Batch)
