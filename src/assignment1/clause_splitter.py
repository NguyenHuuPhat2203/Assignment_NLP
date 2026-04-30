"""
Clause Splitter — Assignment 1, Task 1.1

Splits legal contract text into clauses using:
- spaCy sentence segmentation
- Rule-based splitting on SCONJ/CC dependency edges
- Keyword detection: if, unless, provided that, whereas, subject to

Input:  input/raw_contracts.txt
Output: output/clauses.txt  (one clause per line, blank lines preserved between sections)
"""

from __future__ import annotations

import re
from pathlib import Path

import spacy
from spacy.tokens import Doc, Span

PROJECT_ROOT = Path(__file__).parent.parent.parent

CONDITIONAL_KEYWORDS: frozenset[str] = frozenset({
    "if", "unless", "provided", "whereas", "subject",
    "upon", "when", "whenever", "although", "though",
    "because", "since", "after", "before", "until",
})

COORD_KEYWORDS: frozenset[str] = frozenset({"and", "but", "or", "nor", "yet", "so"})

_SUBORDINATE_DEPS: frozenset[str] = frozenset({
    "ccomp", "relcl", "acl", "xcomp", "advcl",
    "pcomp", "acl:relcl", "compound", "appos",
})


def _has_subject_and_verb(span: Span) -> bool:
    """Return True if *span* contains a non-subordinate subject-verb pair.

    Checks that there exists a nsubj (or equivalent) token whose immediate
    syntactic head is inside the span and carries a main-clause dependency
    label (not inside a relative/complement/subordinate clause).
    """
    span_indices = {tok.i for tok in span}
    for tok in span:
        if tok.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"}:
            head = tok.head
            if head.i in span_indices and head.dep_ not in _SUBORDINATE_DEPS:
                return True
    return False


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_on_coord(sent: Span) -> list[str]:
    """
    Split a sentence on CC tokens only when both sides form independent clauses
    (each side must have a non-subordinate subject-verb pair).
    Keeps conditional constructs intact.
    """
    tokens = list(sent)
    split_indices: list[int] = []

    for i, tok in enumerate(tokens):
        if tok.dep_ == "cc" and tok.text.lower() in COORD_KEYWORDS:
            left = sent.doc[sent.start : sent.start + i]
            right = sent.doc[sent.start + i + 1 : sent.end]
            if _has_subject_and_verb(left) and _has_subject_and_verb(right):
                split_indices.append(i)

    if not split_indices:
        return [_clean(sent.text)]

    parts: list[str] = []
    prev = 0
    for idx in split_indices:
        part = _clean(sent.doc[sent.start + prev : sent.start + idx].text)
        if part:
            parts.append(part)
        prev = idx + 1
    tail = _clean(sent.doc[sent.start + prev : sent.end].text)
    if tail:
        parts.append(tail)
    return parts


def _split_on_sconj(sent: Span) -> list[str] | None:
    """
    Detect subordinating conjunctions that introduce conditional/adverbial clauses.
    When found, the entire sentence is returned as a single clause so that the
    conditional clause and its main clause are never separated.

    Returns the sentence as a one-element list, or None if no SCONJ is detected.
    """
    for tok in sent:
        if tok.dep_ == "mark" and tok.text.lower() in CONDITIONAL_KEYWORDS:
            return [_clean(sent.text)]
    return None


def split_into_clauses(text: str, nlp: spacy.Language | None = None) -> list[str]:
    """Split a legal contract text into individual clauses.

    Sentences that contain a subordinating conjunction (if, unless, provided
    that, whereas …) are kept intact as a single clause.  Other sentences are
    further split at coordinating conjunctions only when both sides are
    independently complete (each side has a subject and a finite verb).

    Args:
        text: Raw contract text.
        nlp:  Optional pre-loaded spaCy model; loaded automatically when None.

    Returns:
        List of clause strings, one clause per element.  Short fragments
        (≤ 2 tokens) are discarded.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    clauses: list[str] = []
    for para in paragraphs:
        doc: Doc = nlp(para)
        for sent in doc.sents:
            sent_text = _clean(sent.text)
            if not sent_text:
                continue
            sconj_result = _split_on_sconj(sent)
            if sconj_result is not None:
                clauses.extend(c for c in sconj_result if c)
                continue
            parts = _split_on_coord(sent)
            clauses.extend(c for c in parts if c)

    return [c for c in clauses if len(c.split()) > 2]


def main(output_dir: str | None = None) -> None:
    input_path = PROJECT_ROOT / "input" / "raw_contracts.txt"
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    output_path = _out / "clauses.txt"

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Reading {input_path} ...")
    text = input_path.read_text(encoding="utf-8")

    print("Splitting into clauses...")
    clauses = split_into_clauses(text, nlp)

    output_path.write_text("\n".join(clauses), encoding="utf-8")
    print(f"Wrote {len(clauses)} clauses → {output_path}")


if __name__ == "__main__":
    main()
