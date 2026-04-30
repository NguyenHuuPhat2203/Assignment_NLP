"""
Dependency Parser — Assignment 1, Task 1.3

Runs spaCy dependency parsing on each clause.

Input:  output/clauses.txt
Output: output/dependency.json
    Format: JSON array of clause objects, each with:
    {
        "clause_id": int,
        "text": str,
        "tokens": [
            {
                "id": int,
                "token": str,
                "lemma": str,
                "pos": str,
                "tag": str,
                "dep": str,
                "head_id": int,
                "head_token": str
            }, ...
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import spacy
from spacy.tokens import Doc

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_clause(clause: str, clause_id: int, nlp: spacy.Language) -> dict:
    """Parse a single clause with spaCy and extract dependency information.

    Args:
        clause:    Clause text.
        clause_id: Zero-based index for this clause.
        nlp:       Loaded spaCy model.

    Returns:
        Dict with keys clause_id, text, and tokens (list of token dicts).
    """
    doc: Doc = nlp(clause)
    tokens = [
        {
            "id": tok.i,
            "token": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "tag": tok.tag_,
            "dep": tok.dep_,
            "head_id": tok.head.i,
            "head_token": tok.head.text,
        }
        for tok in doc
    ]
    return {"clause_id": clause_id, "text": clause, "tokens": tokens}


def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    clauses_path = Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    output_path = _out / "dependency.json"

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Reading {clauses_path} ...")
    clauses = [
        ln.strip()
        for ln in clauses_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    print(f"Parsing {len(clauses)} clauses...")
    results = [parse_clause(clause, i, nlp) for i, clause in enumerate(clauses)]

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote dependency parse for {len(results)} clauses → {output_path}")


if __name__ == "__main__":
    main()
