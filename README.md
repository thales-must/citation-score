# DCRE

## Python Libraries

Python 3.13.9

```bash
pip install -r requirements.txt
```

## Set Up Environment Variables

Rename `.env.local` to `.env`.

### 1. Basic Information

| Parameter | Description |
|---|---|
| PROJECT_NAME | Project name |
| SERVER_DIR | Docker data directory |
| SEED | Random seed for reproducibility |

### 2. Database

| Parameter | Description |
|---|---|
| POSTGRES_USER | Username |
| POSTGRES_PASSWORD | Password |
| POSTGRES_DB | Database name |
| POSTGRES_HOST | Database host |
| POSTGRES_PORT | Port |

### 3. Large Language Model

| Parameter | Description |
|---|---|
| LLM_KEY | LLM API key |
| LLM_URL | LLM endpoint |
| LLM_MODEL | Model |

### 4. Scholarly Database

| Parameter | Description |
|---|---|
| API_KEY | Providing an API key allows requests to be made at shorter intervals. |

You can apply for a key on the [Semantic Scholar website](https://www.semanticscholar.org).

## Seed Paper Collection

A database connection and the **Semantic Scholar API** are required.

File: `collect.ipynb`

Collect all papers published in 2024.

> The Semantic Scholar API does not provide an endpoint for the specific query we need. Therefore, the data must first be saved, after which the database can be used to retrieve the required seed papers.

Select 320 papers across 8 categories, with 40 papers per category. Within each category, rank the 40 papers by cross-encoder score and divide them into 4 stages of 10 papers each.

## Author Synchronization

Synchronize paper author information and import the authors’ previously published papers.

File: `author.ipynb`

> Importing author information also imports papers published by those authors. Many of these papers were published before 2024, which allows us to check whether the authors have collaborated previously.

## Seed Paper Citation Coverage Analysis

File: `data_sampling.ipynb`

## Citation Scoring

Analyze the relevance of all records that contain citation contexts.

File: `relevance.ipynb`

> See the [Sentence Transformers website](https://sbert.net/).

## Paper Selection

Select 320 papers across 8 categories, with 40 papers per category. Within each category, rank the 40 papers by cross-encoder score and divide them into 4 stages of 10 papers each.

File: `subset.ipynb`

## Expert Labeling

Have experts label the selected papers for use in subsequent experiments.

Expert labeling file: `expert_label.xlsx`

The `label.ipynb` notebook uses the expert labels to generate bar charts and heat maps.

## Validation Against Expert Labels

Run LLM scoring on all papers labeled by experts.

File: *(not specified)*

- Validate the embedding model
- Select the LLM
- Determine the weights
- Determine the penalty coefficient for missing citation context

File: `expert_evaluation.ipynb`

## Rescoring

Rescore the papers using their full-text PDFs, and generate figures and tables.

File: `dao_rescore.ipynb`

> When a special sample is identified, find the relevant paper PDF manually.

## Independence

Use expert information to determine one-hop and two-hop relationships, then calculate author independence scores.

The information is recorded in `survey.xlsx`.

After determining the independence coefficient, the full dataset can be processed:

```bash
python .\score_all_citations.py --write --max-llm-calls 10000
```

> This updates the database. Each record stores the cross-encoder score, the LLM score, the abstract-based score when citation context is missing, and the author independence score.

To review all author independence-related parameters, see `author_independence.ipynb`.

## Overall Score Experiment

Compare the overall scores and citation counts across the full dataset.

File: `count_score.ipynb`

This notebook can be used to view an overall bar chart, examine the difference between citation counts and total scores, and inspect the scoring process for an individual paper.