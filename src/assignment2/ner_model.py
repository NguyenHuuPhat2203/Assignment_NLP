"""
Named Entity Recognition — Assignment 2, Task 2.1

Fine-tunes Legal-BERT (nlpaueb/legal-bert-base-uncased) for NER on legal contracts.

Entity schema:
    PARTY   - contracting parties (Party A, Employer, Employee, Lessor, Lessee)
    MONEY   - monetary amounts ($5,000, 10,000 VND, USD 1,000)
    DATE    - dates and time expressions (05 May 2024, within 30 days)
    RATE    - interest/penalty rates (1% per day, 12% per annum)
    PENALTY - penalty clauses
    LAW     - legal references (Civil Code Article 10, Law No. 91/2015)

Usage:
    python src/assignment2/ner_model.py  
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

LABEL_LIST = ["O", "B-PARTY", "I-PARTY", "B-MONEY", "I-MONEY", "B-DATE", "I-DATE",
              "B-RATE", "I-RATE", "B-PENALTY", "I-PENALTY", "B-LAW", "I-LAW"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}



class RuleBasedNER:
    """Regex-based NER for MONEY, DATE, RATE, PARTY, and LAW entities."""

    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("MONEY", re.compile(
            r"\b(?:USD|VND|EUR|GBP|AUD)[\s]?\d[\d,\.]*"
            r"|\$[\d,\.]+(?:\s*(?:USD|VND|EUR))?"
            r"|\d[\d,\.]*\s*(?:USD|VND|EUR|GBP)",
            re.IGNORECASE,
        )),
        ("RATE", re.compile(
            r"\d+(?:\.\d+)?%\s*(?:per\s+(?:day|month|annum|year|week|quarter))",
            re.IGNORECASE,
        )),
        ("DATE", re.compile(
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
            r"|September|October|November|December)\s+\d{4}"
            r"|\b(?:January|February|March|April|May|June|July|August"
            r"|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
            r"|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
            r"|\bwithin\s+\d+\s+(?:days?|months?|years?|weeks?|business\s+days?)"
            r"|\b\d+\s+(?:days?|months?|years?|weeks?)\s+(?:of|from|after|before|notice)",
            re.IGNORECASE,
        )),
        ("PARTY", re.compile(
            r"\bParty\s+[A-Z]\b"
            r"|\b(?:Employer|Employee|Lessor|Lessee|Buyer|Seller|Licensor|Licensee"
            r"|Borrower|Lender|Franchisor|Franchisee|Contractor|Consultant|Tenant"
            r"|Landlord|Assignor|Assignee|Creditor|Guarantor|Obligor|Obligee"
            r"|Distributor|Manufacturer|Agent|Principal|Service\s+Provider)\b",
            re.IGNORECASE,
        )),
        ("LAW", re.compile(
            r"\b(?:Civil\s+Code|Labor\s+Code|Law\s+on\s+\w+(?:\s+\w+)?)"
            r"(?:\s+(?:Article|Section|No\.?)\s+[\w\/\.]+)?"
            r"|\bLaw\s+No\.?\s+[\d\/A-Z]+",
            re.IGNORECASE,
        )),
        ("PENALTY", re.compile(
            r"\b(?:late\s+payment\s+)?penalty\b",
            re.IGNORECASE,
        )),
    ]

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Return list of entity dicts from a text string."""
        entities: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []

        for label, pattern in self._PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if any(s <= start < e or s < end <= e for s, e in occupied):
                    continue
                entities.append({"text": text[start:end], "label": label,
                                  "start": start, "end": end})
                occupied.append((start, end))

        entities.sort(key=lambda x: x["start"])
        return entities


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _char_to_token_labels(
    tokenizer: Any,
    text: str,
    char_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert character-level entity spans to token-level BIO labels.

    Returns a dict compatible with HuggingFace ``Dataset``.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]
    word_ids: list[int | None] = encoding.word_ids()

    labels = ["O"] * len(offset_mapping)

    for ent in char_entities:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        entity_started = False   # True once the first word of this entity is labeled
        prev_word_id: int | None = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0:
                labels[token_idx] = "O"
                continue
            if char_start >= ent_start and char_end <= ent_end:
                wid = word_ids[token_idx]
                if wid != prev_word_id:
                    # First subword of a new word inside the entity
                    labels[token_idx] = f"B-{ent_label}" if not entity_started else f"I-{ent_label}"
                    entity_started = True
                    prev_word_id = wid
                else:
                    # Continuation subword of the same word
                    labels[token_idx] = f"I-{ent_label}"

    label_ids = [LABEL2ID.get(lbl, LABEL2ID["O"]) for lbl in labels]
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_ids,
    }


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class LegalNERModel:
    """Fine-tuned Legal-BERT NER model for legal contract entities."""

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased") -> None:
        self.model_name = model_name
        self.tokenizer: Any = None
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        logger.info("Loading tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    def load_training_data(self, json_path: str) -> list[dict[str, Any]]:
        """Load annotated training examples from a JSON file."""
        with open(json_path, encoding="utf-8") as fh:
            data: list[dict[str, Any]] = json.load(fh)
        logger.info("Loaded %d training examples from %s", len(data), json_path)
        return data

    def prepare_dataset(self, examples: list[dict[str, Any]]) -> Any:
        """Convert char-level annotation spans to a HuggingFace ``Dataset``."""
        from datasets import Dataset

        records: list[dict[str, Any]] = []
        for ex in examples:
            record = _char_to_token_labels(
                self.tokenizer, ex["text"], ex.get("entities", [])
            )
            records.append(record)

        return Dataset.from_list(records)

    def train(
        self,
        train_data: list[dict[str, Any]],
        save_path: str = "models/ner_legal_bert",
    ) -> None:
        """Fine-tune the model on training data and save to *save_path*."""
        from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

        save_dir = PROJECT_ROOT / save_path
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.prepare_dataset(train_data)

        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        args = TrainingArguments(
            output_dir=str(save_dir),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            no_cuda=not _cuda_available(),
            push_to_hub=False,
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting NER fine-tuning …")
        trainer.train()
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info("Model saved to %s", save_dir)

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq: list[str] = []
            pred_seq_str: list[str] = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_seq.append(ID2LABEL[int(l)])
                pred_seq_str.append(ID2LABEL[int(p)])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_str)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Run NER inference on a single text string.

        Returns a list of ``{"text": str, "label": str, "start": int, "end": int}``.
        """
        import torch

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping: list[tuple[int, int]] = encoding.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()
        pred_labels = [ID2LABEL[pid] for pid in pred_ids]

        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for token_idx, (label, (char_start, char_end)) in enumerate(
            zip(pred_labels, offset_mapping)
        ):
            if char_start == 0 and char_end == 0:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": text[char_start:char_end],
                    "label": label[2:],
                    "start": char_start,
                    "end": char_end,
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == label[2:]:
                current_entity["text"] = text[current_entity["start"]:char_end]
                current_entity["end"] = char_end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def evaluate(self, test_data: list[dict[str, Any]]) -> dict[str, float]:
        """Compute seqeval F1/Precision/Recall on annotated test examples."""
        from seqeval.metrics import f1_score, precision_score, recall_score

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []

        for ex in test_data:
            text = ex["text"]
            gold_entities = ex.get("entities", [])

            true_bio = _bio_from_char_spans(text, gold_entities)
            pred_entities = self.predict(text)
            pred_bio = _bio_from_char_spans(text, pred_entities)

            min_len = min(len(true_bio), len(pred_bio))
            true_labels.append(true_bio[:min_len])
            pred_labels.append(pred_bio[:min_len])

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bio_from_char_spans(text: str, entities: list[dict[str, Any]]) -> list[str]:
    """Produce a character-level BIO tag sequence from entity spans."""
    tags = ["O"] * len(text)
    for ent in entities:
        label = ent["label"]
        start, end = ent["start"], ent["end"]
        for i in range(start, end):
            if i >= len(tags):
                break
            tags[i] = f"B-{label}" if i == start else f"I-{label}"
    return tags


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    training_json = PROJECT_ROOT / "data" / "english" / "sample_ner_training.json"
    _clauses_file = Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    ner_output = _out / "ner_results.json"

    if not _clauses_file.exists():
        logger.error("clauses.txt not found at %s — run Assignment 1 first.", _clauses_file)
        return

    clauses: list[str] = [
        line.strip() for line in _clauses_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d clauses from %s", len(clauses), _clauses_file)

    rule_ner = RuleBasedNER()
    results: list[dict[str, Any]] = []

    try:
        model = LegalNERModel()
        train_data = model.load_training_data(str(training_json))
        model.train(train_data, save_path="models/ner_legal_bert")
        logger.info("Using fine-tuned Legal-BERT for prediction.")
        predict_fn = model.predict
    except Exception as exc:
        logger.warning("BERT NER training failed (%s); falling back to rule-based NER.", exc)
        predict_fn = rule_ner.predict

    for clause_id, clause in enumerate(clauses):
        entities = predict_fn(clause)
        results.append({"clause_id": clause_id, "text": clause, "entities": entities})

    with open(ner_output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    logger.info("NER results written to %s (%d clauses).", ner_output, len(results))


if __name__ == "__main__":
    main()
