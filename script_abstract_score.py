#!/usr/bin/env python3
"""
Compute abstract-only citation-relevance scores for an expert-labeled XLSX file.

The script intentionally excludes ``citation_context`` from both scorers.
It uses:

1. CitationRelevance.score_cross(citing_abstract, cited_abstract)
2. GPT-5-mini with citing_abstract and cited_abstract only

The output is checkpointed atomically and can be resumed by running the same
command again. Rows are later merged with other tables by ``instance_id``.

Run directly with the default filenames:

    python -u score_abstract_only.py

Test the first three rows:

    python -u score_abstract_only.py --limit 3

Required environment variables:

    LLM_KEY=...
    LLM_URL=...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm.auto import tqdm

from scholar import CitationRelevance

# Fixed experiment settings. Normally, only INPUT_XLSX needs to be changed.
INPUT_XLSX = "expert_label.xlsx"
OUTPUT_XLSX = "sample/abstract_only_scores.xlsx"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GPT_MODEL = "gpt-5-mini"
DEVICE = "cpu"
CHECKPOINT_EVERY = 5
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 2.0
REQUEST_INTERVAL = 0.0

SYSTEM_PROMPT = "You are a careful and neutral academic reviewer."


def clean_text(value: Any) -> str:
    """Normalize an Excel cell to a compact plain-text string."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def excel_safe_text(value: Any, limit: int = 30_000) -> str:
    """Keep diagnostic text below Excel's per-cell text limit."""
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_prompt(citing_abstract: str, cited_abstract: str) -> str:
    """Build an abstract-only prompt with the same four score intervals."""
    return f"""
Evaluate the relevance of the cited paper to the citing paper using only the
two abstracts below. This simulates a citation for which the local in-text
citation context is unavailable.

Do not assume access to the citation sentence or surrounding text.
Do not infer a specific citation function that is not supported by the abstracts.
Do not assign a high score merely because the papers belong to the same field.

Use a continuous score from 0.00 to 1.00:
- 0.00--0.24: no meaningful relationship or only incidental overlap;
- 0.25--0.49: broad topical/background relationship;
- 0.50--0.74: direct and substantive methodological, theoretical, or empirical relationship;
- 0.75--1.00: very strong relevance or likely central dependence supported by the abstracts.

Return one JSON object only:
{{
  "score": <number from 0.00 to 1.00>,
  "reason": "<brief justification based only on the abstracts>"
}}

Citing paper abstract:
{citing_abstract}

Cited paper abstract:
{cited_abstract}
""".strip()


def parse_json_object(content: str) -> Dict[str, Any]:
    """Parse a JSON response, tolerating an accidental Markdown code fence."""
    content = (content or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("The LLM response is not a JSON object.")

    return payload


def request_llm_score(
    client: Any,
    citing_abstract: str,
    cited_abstract: str,
) -> Dict[str, Any]:
    """Call GPT-5-mini with exponential backoff and validate its score."""
    prompt = build_prompt(citing_abstract, cited_abstract)
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            payload = parse_json_object(content)
            score = float(payload["score"])

            if not 0.0 <= score <= 1.0:
                raise ValueError(f"LLM score is outside [0, 1]: {score}")

            return {
                "score": score,
                "reason": excel_safe_text(payload.get("reason", "")),
                "raw": excel_safe_text(content),
                "error": "",
                "model": GPT_MODEL,
                "scored_at": utc_now(),
            }

        except Exception as exc:  # API/proxy errors vary by provider.
            last_error = exc
            if attempt + 1 >= MAX_RETRIES:
                break

            delay = RETRY_BASE_SECONDS * (2**attempt)
            delay += random.uniform(0.0, min(1.0, RETRY_BASE_SECONDS))
            tqdm.write(
                f"LLM request failed ({attempt + 1}/{MAX_RETRIES}); "
                f"retrying in {delay:.1f}s: {exc}"
            )
            time.sleep(delay)

    return {
        "score": None,
        "reason": "",
        "raw": "",
        "error": excel_safe_text(repr(last_error)),
        "model": GPT_MODEL,
        "scored_at": utc_now(),
    }



def atomic_save_xlsx(df: pd.DataFrame, output_path: Path) -> None:
    """Write a complete checkpoint and atomically publish it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp.xlsx")

    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="abstract_only_scores", index=False)

        sheet = writer.book["abstract_only_scores"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    os.replace(temp_path, output_path)


def prepare_result_table(source: pd.DataFrame) -> pd.DataFrame:
    """Create a merge-friendly output table without citation_context."""
    identity_columns = [
        "instance_id",
        "field",
        "citing_corpus_id",
        "citing_title",
        "citing_abstract",
        "cited_corpus_id",
        "cited_title",
        "cited_abstract",
    ]
    result = source[[c for c in identity_columns if c in source.columns]].copy()

    result["cross_abstract_score"] = pd.NA
    result["cross_abstract_error"] = ""
    result["cross_abstract_model"] = ""
    result["cross_abstract_scored_at"] = ""
    result["gpt_5_mini_abstract_score"] = pd.NA
    result["gpt_5_mini_abstract_reason"] = ""
    result["gpt_5_mini_abstract_raw"] = ""
    result["gpt_5_mini_abstract_error"] = ""
    result["gpt_5_mini_abstract_model"] = ""
    result["gpt_5_mini_abstract_scored_at"] = ""
    return result


def restore_checkpoint(result: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Restore score columns from an existing output using instance_id."""
    if not output_path.exists():
        return result

    previous = pd.read_excel(output_path, sheet_name="abstract_only_scores")
    if "instance_id" not in previous.columns:
        raise ValueError(f"Checkpoint has no instance_id column: {output_path}")
    if previous["instance_id"].astype(str).duplicated().any():
        raise ValueError(f"Checkpoint contains duplicate instance_id values: {output_path}")

    previous = previous.copy()
    previous["__merge_key"] = previous["instance_id"].astype(str)
    previous = previous.set_index("__merge_key")
    keys = result["instance_id"].astype(str)

    score_columns = [
        c
        for c in result.columns
        if c.startswith("cross_abstract_")
        or c.startswith("gpt_5_mini_abstract_")
    ]

    for column in score_columns:
        if column in previous.columns:
            result[column] = keys.map(previous[column])

    return result


def validate_source(source: pd.DataFrame) -> None:
    required = {"instance_id", "citing_abstract", "cited_abstract"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Input XLSX is missing required columns: {missing}")
    if source["instance_id"].isna().any():
        raise ValueError("Input XLSX contains an empty instance_id.")
    if source["instance_id"].astype(str).duplicated().any():
        duplicates = source.loc[
            source["instance_id"].astype(str).duplicated(keep=False),
            "instance_id",
        ].astype(str).tolist()
        raise ValueError(f"Input XLSX contains duplicate instance_id values: {duplicates[:10]}")


def run_cross_scoring(
    result: pd.DataFrame,
    output_path: Path,
) -> None:
    pending = result.index[pd.to_numeric(
        result["cross_abstract_score"], errors="coerce"
    ).isna()].tolist()

    if not pending:
        print("Cross-encoder: no pending rows.", flush=True)
        return

    scorer = CitationRelevance(
        cross_model=CROSS_MODEL,
        device=DEVICE,
        default_context_agg="max",
    )

    for completed, index in enumerate(
        tqdm(pending, desc="Cross-encoder", unit="row", dynamic_ncols=True),
        start=1,
    ):
        citing_abstract = clean_text(result.at[index, "citing_abstract"])
        cited_abstract = clean_text(result.at[index, "cited_abstract"])

        try:
            if not citing_abstract or not cited_abstract:
                raise ValueError("Missing citing_abstract or cited_abstract.")

            score = float(scorer.score_cross(citing_abstract, cited_abstract))
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Cross-encoder score is outside [0, 1]: {score}")

            result.at[index, "cross_abstract_score"] = score
            result.at[index, "cross_abstract_error"] = ""
            result.at[index, "cross_abstract_model"] = CROSS_MODEL
            result.at[index, "cross_abstract_scored_at"] = utc_now()
        except Exception as exc:
            result.at[index, "cross_abstract_error"] = excel_safe_text(repr(exc))
            result.at[index, "cross_abstract_model"] = CROSS_MODEL
            result.at[index, "cross_abstract_scored_at"] = utc_now()

        if completed % CHECKPOINT_EVERY == 0:
            atomic_save_xlsx(result, output_path)

    atomic_save_xlsx(result, output_path)


def run_llm_scoring(
    result: pd.DataFrame,
    output_path: Path,
) -> None:
    pending = result.index[pd.to_numeric(
        result["gpt_5_mini_abstract_score"], errors="coerce"
    ).isna()].tolist()

    if not pending:
        print("GPT-5-mini: no pending rows.", flush=True)
        return

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for GPT scoring. "
            "Install it with: pip install -U openai"
        ) from exc

    api_key = os.getenv("LLM_KEY")
    base_url = os.getenv("LLM_URL")
    if not api_key:
        raise EnvironmentError("Environment variable LLM_KEY is not set.")
    if not base_url:
        raise EnvironmentError("Environment variable LLM_URL is not set.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for completed, index in enumerate(
        tqdm(pending, desc="GPT-5-mini", unit="row", dynamic_ncols=True),
        start=1,
    ):
        citing_abstract = clean_text(result.at[index, "citing_abstract"])
        cited_abstract = clean_text(result.at[index, "cited_abstract"])

        if not citing_abstract or not cited_abstract:
            llm_result = {
                "score": None,
                "reason": "",
                "raw": "",
                "error": "Missing citing_abstract or cited_abstract.",
                "model": GPT_MODEL,
                "scored_at": utc_now(),
            }
        else:
            llm_result = request_llm_score(
                client=client,
                citing_abstract=citing_abstract,
                cited_abstract=cited_abstract,
            )

        result.at[index, "gpt_5_mini_abstract_score"] = llm_result["score"]
        result.at[index, "gpt_5_mini_abstract_reason"] = llm_result["reason"]
        result.at[index, "gpt_5_mini_abstract_raw"] = llm_result["raw"]
        result.at[index, "gpt_5_mini_abstract_error"] = llm_result["error"]
        result.at[index, "gpt_5_mini_abstract_model"] = llm_result["model"]
        result.at[index, "gpt_5_mini_abstract_scored_at"] = llm_result["scored_at"]

        if completed % CHECKPOINT_EVERY == 0:
            atomic_save_xlsx(result, output_path)

        if REQUEST_INTERVAL > 0:
            time.sleep(REQUEST_INTERVAL)

    atomic_save_xlsx(result, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-encoder and GPT-5-mini scores from two abstracts only."
    )
    parser.add_argument(
        "--input",
        default=INPUT_XLSX,
        help=f"Expert-labeled source XLSX (default: {INPUT_XLSX}).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_XLSX,
        help=f"Output/checkpoint XLSX (default: {OUTPUT_XLSX}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows, useful for a small test run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output checkpoint and start from the input again.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("--output must be different from --input.")

    source = pd.read_excel(input_path, sheet_name=0)
    validate_source(source)

    if args.limit is not None:
        source = source.head(args.limit).copy()

    result = prepare_result_table(source)
    if output_path.exists() and not args.overwrite:
        result = restore_checkpoint(result, output_path)

    # Publish the initialized table before the slow model calls begin.
    atomic_save_xlsx(result, output_path)

    run_cross_scoring(result, output_path)
    run_llm_scoring(result, output_path)

    atomic_save_xlsx(result, output_path)

    cross_done = pd.to_numeric(
        result["cross_abstract_score"], errors="coerce"
    ).notna().sum()
    llm_done = pd.to_numeric(
        result["gpt_5_mini_abstract_score"], errors="coerce"
    ).notna().sum()

    print(
        f"Completed. rows={len(result)}, cross_scored={cross_done}, "
        f"llm_scored={llm_done}, output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()