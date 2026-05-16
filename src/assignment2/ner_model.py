from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

LABEL_LIST = [
    "O",
    "B-PARTY",
    "I-PARTY",
    "B-MONEY",
    "I-MONEY",
    "B-DATE",
    "I-DATE",
    "B-RATE",
    "I-RATE",
    "B-PENALTY",
    "I-PENALTY",
    "B-LAW",
    "I-LAW",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

_MONTHS = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
)


class RuleBasedNER:
    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "MONEY",
            re.compile(
                r"\b(?:USD|VND|EUR|GBP|AUD)\s*[\d,\.]+"
                r"|\$\s*[\d,\.]+(?:\s*(?:USD|VND|EUR|dollars?))?"
                r"|\b[\d,\.]+\s*(?:USD|VND|EUR|GBP|dollars?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "RATE",
            re.compile(
                r"\b\d+(?:\.\d+)?%",
                re.IGNORECASE,
            ),
        ),
        (
            "DATE",
            re.compile(
                r"\b\d{1,2}(?:st|nd|rd|th)\s+day\s+of\s+(?:" + _MONTHS + r"),?\s+\d{4}"
                r"|\b\d{1,2}\s+(?:" + _MONTHS + r")\s+\d{4}"
                r"|\b(?:" + _MONTHS + r")\s+\d{1,2},?\s+\d{4}"
                r"|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
                r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten"
                r"|eleven|twelve|fifteen|twenty|thirty|forty|forty-five|sixty|ninety"
                r")\s+\(\d+\)\s+(?:consecutive\s+)?(?:Business\s+)?(?:days?|months?|years?|weeks?)"
                r"|\b\d+\s+\(\w+\)\s+(?:consecutive\s+)?(?:Business\s+)?(?:days?|months?|years?|weeks?)"
                r"|\bwithin\s+\d+\s+(?:Business\s+)?(?:days?|months?|years?|weeks?)"
                r"|\b\d+\s+(?:consecutive\s+)?(?:Business\s+)?(?:days?|months?|years?|weeks?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PARTY",
            re.compile(
                r"\bParty\s+[A-Z]\b"
                r"|\bthe\s+Parties\b"
                r"|\b(?:Employer|Employee|Lessor|Lessee|Buyer|Seller|Licensor|Licensee"
                r"|Borrower|Lender|Franchisor|Franchisee|Contractor|Consultant|Tenant"
                r"|Landlord|Assignor|Assignee|Creditor|Guarantor|Obligor|Obligee"
                r"|Distributor|Manufacturer|Principal|Service\s+Provider|Client"
                r"|Indemnitor|Indemnified\s+Party|Affected\s+Party)\b",
            ),
        ),
        (
            "LAW",
            re.compile(
                r"\bLaw\s+No\.?\s+[\d\/\w\-]+"
                r"|\bDecree\s+No\.?\s+[\d\/\w\-]+"
                r"|\bCircular\s+No\.?\s+[\d\/\w\-]+"
                r"|\b(?:Civil|Labor|Commercial)\s+(?:Code|Law)(?:\s+No\.?\s+[\d\/\w\-]+)?"
                r"(?:\s+Article\s+\d+)?"
                r"|\bLaw\s+on\s+[\w\s\-]+(?=\s)"
                r"|\b[\w\s]+(?:Protection|Liability|Anti-Corruption|Anti-Bribery)\s+Act"
                r"(?:\s+Article\s+\d+)?"
                r"|\bUnited\s+Nations\s+Convention\s+against\s+Corruption"
                r"|\bSIAC\s+Rules",
                re.IGNORECASE,
            ),
        ),
        (
            "PENALTY",
            re.compile(
                r"\b(?:late\s+payment\s+)?penalty\b"
                r"|\btermination\s+fee\b"
                r"|\bliquidated\s+damages?\b"
                r"|\bpunitive\s+damages?\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def predict(self, text: str) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []

        for label, pattern in self._PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if any(s <= start < e or s < end <= e for s, e in occupied):
                    continue
                entities.append(
                    {
                        "text": text[start:end],
                        "label": label,
                        "start": start,
                        "end": end,
                    }
                )
                occupied.append((start, end))

        entities.sort(key=lambda x: x["start"])
        return entities


_STOPWORD_SPANS: frozenset[str] = frozenset(
    {
        "of",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "for",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "any",
        "all",
        "such",
        "no",
    }
)


def _is_valid_silver_entity(entity: dict[str, Any]) -> bool:

    text = entity.get("text", "").strip()
    if len(text) < 2:
        return False

    if not any(ch.isalnum() for ch in text):
        return False

    tokens = text.lower().split()
    if tokens and all(tok in _STOPWORD_SPANS for tok in tokens):
        return False
    return True


def _generate_silver_data(
    clauses: list[str], rule_ner: RuleBasedNER
) -> list[dict[str, Any]]:

    silver: list[dict[str, Any]] = []
    n_dropped = 0
    for clause in clauses:
        raw_entities = rule_ner.predict(clause)
        entities = [e for e in raw_entities if _is_valid_silver_entity(e)]
        n_dropped += len(raw_entities) - len(entities)
        if entities:
            silver.append({"text": clause, "entities": entities})
    logger.info(
        "Generated %d silver-label examples from %d clauses (dropped %d noisy spans).",
        len(silver),
        len(clauses),
        n_dropped,
    )
    return silver


def _merge_entities(
    rule_entities: list[dict[str, Any]],
    bert_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    merged = list(rule_entities)
    occupied = [(e["start"], e["end"]) for e in rule_entities]

    for ent in bert_entities:
        s, e = ent["start"], ent["end"]
        if not any(a < e and s < b for a, b in occupied):
            merged.append(ent)
            occupied.append((s, e))

    merged.sort(key=lambda x: x["start"])
    return merged


def _char_to_token_labels(
    tokenizer: Any,
    text: str,
    char_entities: list[dict[str, Any]],
) -> dict[str, Any]:
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
        entity_started = False
        prev_word_id: int | None = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0:
                labels[token_idx] = "O"
                continue
            if char_start >= ent_start and char_end <= ent_end:
                wid = word_ids[token_idx]
                if wid != prev_word_id:
                    labels[token_idx] = (
                        f"B-{ent_label}" if not entity_started else f"I-{ent_label}"
                    )
                    entity_started = True
                    prev_word_id = wid
                else:
                    labels[token_idx] = f"I-{ent_label}"

    label_ids = [LABEL2ID.get(lbl, LABEL2ID["O"]) for lbl in labels]
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_ids,
    }


class LegalNERModel:
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

    @classmethod
    def from_saved(cls, path: str | Path) -> "LegalNERModel":

        from transformers import AutoModelForTokenClassification, AutoTokenizer

        obj = cls.__new__(cls)
        obj.model_name = str(path)
        obj.tokenizer = AutoTokenizer.from_pretrained(str(path))
        obj.model = AutoModelForTokenClassification.from_pretrained(str(path))
        logger.info("Loaded fine-tuned NER model from %s", path)
        return obj

    def load_training_data(self, json_path: str) -> list[dict[str, Any]]:
        with open(json_path, encoding="utf-8") as fh:
            data: list[dict[str, Any]] = json.load(fh)
        logger.info("Loaded %d training examples from %s", len(data), json_path)
        return data

    def prepare_dataset(self, examples: list[dict[str, Any]]) -> Any:
        from datasets import Dataset

        records = [
            _char_to_token_labels(self.tokenizer, ex["text"], ex.get("entities", []))
            for ex in examples
        ]
        return Dataset.from_list(records)

    def train(
        self,
        train_data: list[dict[str, Any]],
        save_path: str = "models/ner_legal_bert",
    ) -> None:
        from transformers import (
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )

        save_dir = PROJECT_ROOT / save_path
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.prepare_dataset(train_data)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        args = TrainingArguments(
            output_dir=str(save_dir),
            num_train_epochs=5,
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
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting NER fine-tuning on %d examples …", len(train_data))
        trainer.train()
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info("Model saved to %s", save_dir)

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        from seqeval.metrics import f1_score, precision_score, recall_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq, pred_seq_str = [], []
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
        import torch

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping: list[tuple[int, int]] = encoding.pop("offset_mapping")[
            0
        ].tolist()

        with torch.no_grad():
            outputs = self.model(**encoding)

        pred_ids = outputs.logits[0].argmax(dim=-1).tolist()
        pred_labels = [ID2LABEL[pid] for pid in pred_ids]

        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for label, (char_start, char_end) in zip(pred_labels, offset_mapping):
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
            elif (
                label.startswith("I-")
                and current_entity
                and current_entity["label"] == label[2:]
            ):
                current_entity["text"] = text[current_entity["start"] : char_end]
                current_entity["end"] = char_end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


def _bio_from_char_spans(text: str, entities: list[dict[str, Any]]) -> list[str]:
    tags = ["O"] * len(text)
    for ent in entities:
        label = ent["label"]
        start, end = ent["start"], ent["end"]
        for i in range(start, min(end, len(tags))):
            tags[i] = f"B-{label}" if i == start else f"I-{label}"
    return tags


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    training_json = PROJECT_ROOT / "data" / "english" / "sample_ner_training.json"
    _clauses_file = (
        Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    )
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    ner_output = _out / "ner_results.json"

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

    rule_ner = RuleBasedNER()

    saved_model_path = PROJECT_ROOT / "models" / "ner_legal_bert_v2"
    bert_predict_fn = None

    if saved_model_path.exists():
        try:
            bert_model = LegalNERModel.from_saved(saved_model_path)
            bert_predict_fn = bert_model.predict
            logger.info("Using saved fine-tuned model for BERT predictions.")
        except Exception as exc:
            logger.warning("Could not load saved model (%s); BERT disabled.", exc)
    else:
        logger.info("No saved model found — training with manual + silver labels.")
        try:
            bert_model = LegalNERModel()

            manual_data: list[dict[str, Any]] = []
            if training_json.exists():
                manual_data = bert_model.load_training_data(str(training_json))

            silver_data = _generate_silver_data(clauses, rule_ner)

            all_training = manual_data + silver_data
            logger.info(
                "Training on %d examples (%d manual + %d silver).",
                len(all_training),
                len(manual_data),
                len(silver_data),
            )

            bert_model.train(all_training, save_path="models/ner_legal_bert_v2")
            bert_predict_fn = bert_model.predict
        except Exception as exc:
            logger.warning("BERT training failed (%s); using rule-based only.", exc)

    results: list[dict[str, Any]] = []
    for clause_id, clause in enumerate(clauses):
        rule_entities = rule_ner.predict(clause)

        if bert_predict_fn is not None:
            try:
                bert_entities = bert_predict_fn(clause)

                bert_entities = [e for e in bert_entities if _is_valid_silver_entity(e)]
                entities = _merge_entities(rule_entities, bert_entities)
            except Exception:
                entities = rule_entities
        else:
            entities = rule_entities

        results.append({"clause_id": clause_id, "text": clause, "entities": entities})

    with open(ner_output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    n_with = sum(1 for r in results if r["entities"])
    total_ents = sum(len(r["entities"]) for r in results)
    logger.info(
        "NER results written to %s (%d clauses, %d with entities, %d total entities).",
        ner_output,
        len(results),
        n_with,
        total_ents,
    )


if __name__ == "__main__":
    main()
