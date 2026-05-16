from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

_TIME_PREPS = frozenset(
    {"before", "after", "on", "within", "by", "until", "upon", "during"}
)
_CONDITION_MARKS = frozenset({"if", "unless", "provided", "subject"})


def _subtree_text(token: Any) -> str:

    tokens_in_subtree = sorted(token.subtree, key=lambda t: t.i)
    return " ".join(t.text for t in tokens_in_subtree)


def _noun_phrase(token: Any) -> str:

    if hasattr(token, "subtree"):
        return _subtree_text(token)
    return token.text


class SRLModel:
    def __init__(self) -> None:
        import spacy

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading en_core_web_sm …")
            from spacy.cli import download

            download("en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")

    def extract_roles(self, clause: str) -> dict[str, Any]:

        if not self._is_processable(clause):
            return {
                "clause": clause,
                "predicate": "",
                "roles": {},
            }

        doc = self._nlp(clause)

        if not any(t.pos_ in {"VERB", "AUX"} for t in doc):
            return {
                "clause": clause,
                "predicate": "",
                "roles": {},
            }

        root = self._find_root(doc)
        if root is None:
            return {
                "clause": clause,
                "predicate": "",
                "roles": {},
            }

        roles: dict[str, str] = {}

        for token in doc:
            dep = token.dep_
            head = token.head

            if head == root:
                if dep == "nsubj":
                    roles.setdefault("Agent", _noun_phrase(token))

                elif dep in {"dobj", "obj"}:
                    roles.setdefault("Theme", _noun_phrase(token))

                elif dep == "iobj":
                    roles.setdefault("Recipient", _noun_phrase(token))

                elif dep == "nsubjpass":
                    roles.setdefault("Theme", _noun_phrase(token))

                elif dep == "prep":
                    prep_text = token.text.lower()
                    if prep_text in _TIME_PREPS:
                        roles.setdefault("Time", _subtree_text(token))
                    elif prep_text == "to":
                        roles.setdefault("Recipient", _subtree_text(token))

                elif dep == "advcl":
                    mark_token = next(
                        (c for c in token.children if c.dep_ == "mark"), None
                    )
                    if mark_token and mark_token.text.lower() in _CONDITION_MARKS:
                        roles.setdefault("Condition", _subtree_text(token))

            if dep == "agent" and head == root:
                for child in token.children:
                    if child.dep_ == "pobj":
                        roles["Agent"] = _noun_phrase(child)

        if "Recipient" not in roles:
            for child in root.children:
                if child.dep_ == "prep" and child.text.lower() == "to":
                    roles["Recipient"] = _subtree_text(child)

        if "Time" not in roles:
            for child in root.children:
                if child.dep_ == "prep" and child.text.lower() in _TIME_PREPS:
                    roles["Time"] = _subtree_text(child)

        return {
            "clause": clause,
            "predicate": root.lemma_,
            "roles": roles,
        }

    def process_clauses(self, clauses: list[str]) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []
        for clause_id, clause in enumerate(clauses):
            if not clause.strip():
                continue
            srl = self.extract_roles(clause)
            results.append(
                {
                    "clause_id": clause_id,
                    "text": clause,
                    "predicate": srl["predicate"],
                    "roles": srl["roles"],
                }
            )
        return results

    @staticmethod
    def _find_root(doc: Any) -> Any | None:

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                return token

        for token in doc:
            if token.pos_ in {"VERB", "AUX"}:
                return token
        return None

    @staticmethod
    def _is_processable(clause: str) -> bool:

        stripped = clause.strip()
        if len(stripped) < 5:
            return False

        letters = [c for c in stripped if c.isalpha()]
        if letters and not any(c.islower() for c in letters):
            return False

        return True


def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    _clauses_file = (
        Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    )
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    srl_output = _out / "srl_results.json"

    if not _clauses_file.exists():
        logger.error(
            "clauses.txt not found at %s — run Assignment 1 first.", _clauses_file
        )
        return

    clauses: list[str] = [
        line.strip()
        for line in _clauses_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d clauses from %s", len(clauses), _clauses_file)

    model = SRLModel()
    results = model.process_clauses(clauses)

    with open(srl_output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    logger.info("SRL results written to %s (%d entries).", srl_output, len(results))


if __name__ == "__main__":
    main()
