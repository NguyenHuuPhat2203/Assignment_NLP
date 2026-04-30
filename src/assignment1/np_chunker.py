"""
Noun Phrase Chunker — Assignment 1, Task 1.2

Extracts noun phrases from each clause and outputs IOB tags.

Input:  output/clauses.txt
Output: output/chunks.txt
    Format: token\tIOB-tag per line
            blank line between clauses
    IOB tags: B-NP (beginning of NP), I-NP (inside NP), O (outside)
"""

from __future__ import annotations

from pathlib import Path

import spacy
from spacy.tokens import Doc

PROJECT_ROOT = Path(__file__).parent.parent.parent


def chunk_clause(clause: str, nlp: spacy.Language) -> list[tuple[str, str]]:
    """Parse a single clause and return token-level IOB tags for noun phrases.

    Args:
        clause: Single clause text.
        nlp:    Loaded spaCy model.

    Returns:
        List of (token_text, iob_tag) tuples where iob_tag ∈ {B-NP, I-NP, O}.
    """
    doc: Doc = nlp(clause)

    iob: dict[int, str] = {}
    for chunk in doc.noun_chunks:
        for i, tok in enumerate(chunk):
            iob[tok.i] = "B-NP" if i == 0 else "I-NP"

    return [(tok.text, iob.get(tok.i, "O")) for tok in doc]


def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    clauses_path = Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    output_path = _out / "chunks.txt"

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Reading {clauses_path} ...")
    clauses = [
        ln.strip()
        for ln in clauses_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    lines: list[str] = []
    for clause in clauses:
        pairs = chunk_clause(clause, nlp)
        for token, tag in pairs:
            lines.append(f"{token}\t{tag}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote IOB tags for {len(clauses)} clauses → {output_path}")


if __name__ == "__main__":
    main()
