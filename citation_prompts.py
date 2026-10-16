"""Shared routine citation-scoring prompts (no model/database imports).

Keep the scoring rubric here, not in individual scripts or notebooks. The
context template matches llm_model_select.ipynb; the abstract template matches
script_abstract_score.py. Full-text arbitration is a different task and is
not replaced by these routine-scoring prompts.
"""

from __future__ import annotations

import json
from string import Template

SYSTEM_PROMPT = "You are a careful and neutral academic reviewer."
CONTEXT_PROMPT_VERSION = "anchored_v1_2026_revision"
ABSTRACT_PROMPT_VERSION = "abstract_only_v1"

CONTEXT_TEMPLATE = Template("""You are evaluating the semantic relevance of one citation instance.

Use only the information provided below.
Do not use citation counts, author identities, affiliations, venue prestige,
or any external knowledge.
Do not assume that every citation is important.
Evaluate the citation according to the explicit relationship demonstrated
in the citation context. If the context is vague, generic, or merely lists
references without a clear relationship, assign a low score.

Citation context:
${context}

Cited paper abstract:
${abstract}

Use the following relevance scale:

0.00--0.24:
No clear relevance, misleading citation, or insufficient contextual evidence.

0.25--0.49:
Weak relevance or broad background citation.

0.50--0.74:
Moderate relevance. The cited paper supports a local claim, concept, method,
comparison, empirical result, or related discussion.

0.75--1.00:
Strong or essential relevance. The cited paper is directly used as a method,
dataset, theory, key empirical evidence, or central foundation.

Return strictly valid JSON:

{
  "score": 0.00,
  "citation_function": "background | method | theory | evidence | comparison | dataset | definition | motivation | unclear",
  "evidence_alignment": "none | weak | partial | direct",
  "reason": "A concise explanation based only on the supplied citation context and cited-paper information."
}

The score must be a real number between 0 and 1.
Do not include text outside the JSON object.""")

ABSTRACT_TEMPLATE = Template("""Evaluate the relevance of the cited paper to the citing paper using only the
two abstracts below.

Do not assume access to the citation sentence or surrounding text.
Do not infer a specific citation function that is not supported by the abstracts.
Do not assign a high score merely because the papers belong to the same field.

Use a continuous score from 0.00 to 1.00:
- 0.00--0.24: no meaningful relationship or only incidental overlap;
- 0.25--0.49: broad topical/background relationship;
- 0.50--0.74: direct and substantive methodological, theoretical, or empirical relationship;
- 0.75--1.00: very strong relevance or likely central dependence supported by the abstracts.

Return one JSON object only:
{
  "score": <number from 0.00 to 1.00>,
  "reason": "<brief justification based only on the abstracts>"
}

Citing paper abstract:
${citing_abstract}

Cited paper abstract:
${cited_abstract}""")


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_{name}")
    return value


def build_scoring_prompt(context: str, abstract: str) -> str:
    """Accept already formatted context text, preserving historical formatting."""
    return CONTEXT_TEMPLATE.substitute(
        context=_require_text(context, "context"),
        abstract=_require_text(abstract, "cited_abstract"))


def build_context_prompt(contexts: str | list, cited_abstract: str) -> str:
    """Number database contexts using the existing scripts' [1], [2] format."""
    if isinstance(contexts, str):
        context_text = contexts
    else:
        context_text = "\n\n".join(
            f"[{index}] {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}"
            for index, item in enumerate(contexts or [], start=1))
    return build_scoring_prompt(context_text, cited_abstract)


def build_abstract_prompt(citing_abstract: str, cited_abstract: str) -> str:
    return ABSTRACT_TEMPLATE.substitute(
        citing_abstract=_require_text(citing_abstract, "citing_abstract"),
        cited_abstract=_require_text(cited_abstract, "cited_abstract"))


def build_prompt(branch: str, texts: list[str], cited_abstract: str) -> str:
    """Explicit branch adapter for the full-cohort fill-missing pipeline."""
    if branch == "context":
        return build_context_prompt(texts, cited_abstract)
    if branch == "abstract":
        if len(texts) != 1:
            raise ValueError("abstract_branch_requires_one_citing_abstract")
        return build_abstract_prompt(texts[0], cited_abstract)
    raise ValueError("unknown_scoring_branch")
