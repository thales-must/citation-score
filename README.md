# DCRE

## Python Environment

Python version:

```text
Python 3.13.9
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Rename `.env.local` to `.env` and configure the following variables.

### 1. General Configuration

| Parameter      | Description                     |
| -------------- | ------------------------------- |
| `PROJECT_NAME` | Project name                    |
| `SERVER_DIR`   | Host directory for Docker data  |
| `SEED`         | Random seed for reproducibility |

### 2. Database Configuration

| Parameter           | Description              |
| ------------------- | ------------------------ |
| `POSTGRES_USER`     | PostgreSQL username      |
| `POSTGRES_PASSWORD` | PostgreSQL password      |
| `POSTGRES_DB`       | PostgreSQL database name |
| `POSTGRES_PORT`     | PostgreSQL port          |

### 3. Large Language Model

| Parameter   | Description      |
| ----------- | ---------------- |
| `LLM_KEY`   | LLM API key      |
| `LLM_URL`   | LLM API base URL |
| `LLM_MODEL` | LLM model name   |

### 4. Scholarly Data Source

| Parameter | Description                                        |
| --------- | -------------------------------------------------- |
| `API_KEY` | Semantic Scholar API key for higher request limits |

An API key can be requested from the [Semantic Scholar website](https://www.semanticscholar.org/).

## Seed Paper Collection

This step requires a database connection and access to the Semantic Scholar API.

Notebook:

```text
collect.ipynb
```

The notebook first collects papers according to the specified publication type and year and stores them in the database.

Seed papers are then selected using stratified sampling.

For each seed paper, citing-paper records are retrieved, while only records containing both citation contexts and abstracts are temporarily retained before database insertion.

> The Semantic Scholar API does not directly support the required stratified sampling procedure.
> Candidate papers must therefore be stored first and subsequently sampled using database queries.

## Author Synchronization

This step synchronizes the author information associated with the collected papers.

Notebook:

```text
author.ipynb
```

## Citation Relevance Scoring

For each seed paper, this step retrieves all citing papers, stores the paper and citation records, and computes citation-level relevance scores.

Notebook:

```text
relevance.ipynb
```

> The semantic scoring implementation is based on [Sentence Transformers](https://sbert.net/).

## Expert-Labeled Subset Construction

This step selects 320 citation instances across eight research fields, with 40 instances from each field.

The normalized cross-encoder scores are divided into four intervals, with an initial target of 10 instances per interval for each field.

Notebook:

```text
subset.ipynb
```

## Expert Annotation

The selected citation instances are independently annotated by experts and used as an external reference in subsequent experiments.

Annotation file:

```text
expert_label.xlsx
```

The following notebook generates the expert-label distribution and inter-annotator agreement heatmap:

```text
label.ipynb
```

## Semantic Model Comparison

This experiment compares the citation-level semantic models and generates score distributions, box plots, summary statistics, Spearman correlations, and correlation heatmaps.

Notebook:

```text
sbert.ipynb
```

## LLM Model Selection

The candidate LLMs are evaluated on the expert-labeled subset.

Scoring notebook:

```text
llm_model_select.ipynb
```

The following notebook computes Spearman's correlation, MAE, and QWK and generates the corresponding distribution and box plots:

```text
plot_llm_select.ipynb
```

## Aggregation Weight Calibration

This experiment evaluates different aggregation weights using Spearman's correlation, MAE, and QWK.

Notebook:

```text
llm_weight.ipynb
```

## Context-Missing Penalty Calibration

The following script computes abstract-only relevance scores between the citing and cited papers:

```text
script_abstract_score.py
```

> The script supports background execution and resumable processing.

Candidate penalty factors are evaluated using Spearman's correlation, MAE, and QWK.

Notebook:

```text
penalty.ipynb
```

## Citation Rescoring

This experiment performs citation rescoring using the full-text PDFs of the citing and cited papers and generates the corresponding figures and tables.

Notebook:

```text
rescore.ipynb
```

## Expert Survey on Author Independence

Expert survey responses are used to determine author-independence scores for one-hop and two-hop coauthorship relationships.

Survey data:

```text
inform.xlsx
```

The following notebook generates the survey-result distributions:

```text
inform.ipynb
```

## Paper-Level Aggregation Experiment

This experiment aggregates citation-level scores and compares the resulting paper-level scores with citation counts.

Notebook:

```text
paper.ipynb
```

The notebook supports:

* Visualizing the overall comparison between citation counts and aggregated scores.
* Examining the relative gap between the two measures.
* Inspecting the cumulative scoring process of an individual paper.


