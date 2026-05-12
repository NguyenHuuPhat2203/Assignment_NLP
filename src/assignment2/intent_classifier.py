"""
Clause Intent Classification — Assignment 2, Task 2.3

Classifies legal clauses into one of four intent categories:
    Obligation           - "shall", "must", "required to", "is obligated"
    Prohibition          - "shall not", "must not", "prohibited", "is forbidden"
    Right                - "may", "is entitled to", "has the right", "can"
    Termination Condition - "terminate", "termination", "expiry", "upon breach"

Two models:
    1. Baseline: TF-IDF (1-3 ngrams) + Logistic Regression
    2. Advanced: Fine-tuned BERT (bert-base-uncased)

Input:  output/clauses.txt
Output: output/intent_classification.txt
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

LABEL2ID: dict[str, int] = {
    "Obligation": 0,
    "Prohibition": 1,
    "Right": 2,
    "Termination Condition": 3,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

INTENT_KEYWORDS: dict[str, list[str]] = {
    "Prohibition": [
        r"shall\s+not\b",
        r"must\s+not\b",
        r"\bis\s+prohibited\b",
        r"\bare\s+prohibited\b",
        r"\bis\s+forbidden\b",
        r"\bnot\s+permitted\b",
        r"\bno\s+party\s+shall\b",
        r"\bshall\s+refrain\b",
    ],
    "Obligation": [
        r"\bshall\b(?!\s+not)",
        r"\bmust\b(?!\s+not)",
        r"\bis\s+required\s+to\b",
        r"\bare\s+required\s+to\b",
        r"\bis\s+obligated\s+to\b",
        r"\bhereby\s+agrees\s+to\b",
        r"\bundertakes\s+to\b",
        r"\bwill\s+be\s+responsible\b",
    ],
    "Right": [
        r"\bmay\b",
        r"\bis\s+entitled\s+to\b",
        r"\bare\s+entitled\s+to\b",
        r"\bhas\s+the\s+right\b",
        r"\bhave\s+the\s+right\b",
        r"\bcan\b",
        r"\bat\s+its\s+discretion\b",
        r"\bat\s+their\s+discretion\b",
        r"\bhas\s+the\s+option\b",
    ],
    "Termination Condition": [
        r"\bterminat\w+\b",
        r"\bexpir\w+\b",
        r"\bupon\s+breach\b",
        r"\bin\s+the\s+event\s+of\s+(?:default|breach|non-payment)\b",
        r"\bshall\s+be\s+void\b",
        r"\bnull\s+and\s+void\b",
        r"\bearly\s+termination\b",
        r"\bwithdrawn?\b",
        r"\bdissolution\b",
    ],
}

_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_KEYWORDS.items()
}

_PRIORITY_ORDER: list[str] = [
    "Prohibition",
    "Termination Condition",
    "Right",
    "Obligation",
]


def weak_label(clause: str) -> str:
    """Assign an intent label using keyword heuristics.

    Returns ``"Unknown"`` when no keyword matches.  Prohibition patterns are
    evaluated before Obligation to correctly classify "shall not" clauses.
    """
    for intent in _PRIORITY_ORDER:
        if any(pat.search(clause) for pat in _COMPILED_PATTERNS[intent]):
            return intent
    return "Unknown"



class TFIDFIntentClassifier:
    """Baseline intent classifier using TF-IDF features and Logistic Regression."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self._pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=5000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                    token_pattern=r"\b[a-zA-Z][a-zA-Z0-9\-']*\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ])
        self._label2id = LABEL2ID
        self._id2label = ID2LABEL

    def fit(self, clauses: list[str], labels: list[str]) -> None:
        """Train the TF-IDF + LR pipeline."""
        label_ids = [self._label2id[lbl] for lbl in labels]
        self._pipeline.fit(clauses, label_ids)
        logger.info("TF-IDF classifier trained on %d examples.", len(clauses))

    def predict(self, clauses: list[str]) -> list[str]:
        """Return predicted intent labels for a list of clauses."""
        pred_ids = self._pipeline.predict(clauses)
        return [self._id2label[int(pid)] for pid in pred_ids]

    def predict_proba(self, clauses: list[str]) -> np.ndarray:
        """Return class probability matrix (n_samples × n_classes)."""
        return self._pipeline.predict_proba(clauses)

    def evaluate(self, clauses: list[str], labels: list[str]) -> dict[str, float]:
        """Compute accuracy and macro-F1 on labelled examples."""
        from sklearn.metrics import accuracy_score, f1_score

        label_ids = [self._label2id[lbl] for lbl in labels]
        pred_ids = self._pipeline.predict(clauses)
        return {
            "accuracy": accuracy_score(label_ids, pred_ids),
            "f1_macro": f1_score(label_ids, pred_ids, average="macro"),
        }

    def save(self, path: str | Path) -> None:
        """Serialize the pipeline to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._pipeline, fh)
        logger.info("TF-IDF classifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFIntentClassifier":
        """Deserialize a previously saved classifier from *path*."""
        instance = cls.__new__(cls)
        with open(path, "rb") as fh:
            instance._pipeline = pickle.load(fh)
        instance._label2id = LABEL2ID
        instance._id2label = ID2LABEL
        return instance



class BERTIntentClassifier:
    """Intent classifier backed by a fine-tuned BERT sequence-classification model."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 4,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.num_labels = num_labels
        logger.info("Loading BERT tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    def prepare_data(
        self,
        clauses: list[str],
        labels: list[str],
    ) -> Any:
        """Tokenize clauses and create a HuggingFace ``Dataset``."""
        from datasets import Dataset

        label_ids = [LABEL2ID[lbl] for lbl in labels]
        encodings = self.tokenizer(
            clauses,
            truncation=True,
            padding=True,
            max_length=256,
        )
        records = [
            {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
                "labels": label_ids[i],
            }
            for i in range(len(clauses))
        ]
        return Dataset.from_list(records)

    def train(
        self,
        train_data: tuple[list[str], list[str]],
        save_path: str = "models/intent_bert",
    ) -> None:
        """Fine-tune BERT on *train_data* (clauses, labels) tuple.

        Uses a class-weighted cross-entropy loss to mitigate the severe class
        imbalance observed in legal-contract intent corpora (Obligation
        typically dominates 80–90\\% of clauses).
        """
        import torch
        import torch.nn as nn
        from transformers import (
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        clauses, labels = train_data
        dataset = self.prepare_data(clauses, labels)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        save_dir = PROJECT_ROOT / save_path
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---- compute class weights (inverse frequency, normalised) -----
        label_ids = [LABEL2ID[lbl] for lbl in labels]
        counts = np.bincount(label_ids, minlength=self.num_labels).astype(np.float64)
        # Avoid division by zero for unseen classes.
        counts = np.where(counts == 0, 1.0, counts)
        weights = counts.sum() / (self.num_labels * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(
            "Intent class counts=%s  -> weights=%s",
            counts.tolist(),
            [round(w, 3) for w in weights.tolist()],
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        args = TrainingArguments(
            output_dir=str(save_dir),
            num_train_epochs=8,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            no_cuda=not _cuda_available(),
            push_to_hub=False,
            logging_steps=10,
            report_to="none",
        )

        class _WeightedTrainer(Trainer):
            def compute_loss(
                _self,
                model: Any,
                inputs: Any,
                return_outputs: bool = False,
                **kwargs: Any,
            ) -> Any:
                labels_t = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(
                    weight=class_weights.to(logits.device)
                )
                loss = loss_fct(
                    logits.view(-1, model.config.num_labels),
                    labels_t.view(-1),
                )
                return (loss, outputs) if return_outputs else loss

        trainer = _WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("Starting BERT intent classification fine-tuning …")
        trainer.train()
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info("BERT intent model saved to %s", save_dir)

    def predict(self, clauses: list[str]) -> list[str]:
        """Return predicted intent labels for *clauses*."""
        import torch

        all_preds: list[str] = []
        batch_size = 16

        for start in range(0, len(clauses), batch_size):
            batch = clauses[start:start + batch_size]
            encoding = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encoding)
            pred_ids = outputs.logits.argmax(dim=-1).tolist()
            all_preds.extend(ID2LABEL[pid] for pid in pred_ids)

        return all_preds

    def predict_with_confidence(self, clauses: list[str]) -> list[tuple[str, float]]:
        """Return (label, confidence) pairs for each clause."""
        import torch
        import torch.nn.functional as F

        results: list[tuple[str, float]] = []
        batch_size = 16

        for start in range(0, len(clauses), batch_size):
            batch = clauses[start:start + batch_size]
            encoding = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encoding)
            probs = F.softmax(outputs.logits, dim=-1)
            for prob_row in probs:
                best_idx = int(prob_row.argmax())
                results.append((ID2LABEL[best_idx], float(prob_row[best_idx])))

        return results

    def evaluate(self, test_data: tuple[list[str], list[str]]) -> dict[str, float]:
        """Compute accuracy and macro-F1 on a labelled test set."""
        from sklearn.metrics import accuracy_score, f1_score

        clauses, labels = test_data
        preds = self.predict(clauses)
        pred_ids = [LABEL2ID.get(p, 0) for p in preds]
        true_ids = [LABEL2ID[lbl] for lbl in labels]
        return {
            "accuracy": accuracy_score(true_ids, pred_ids),
            "f1_macro": f1_score(true_ids, pred_ids, average="macro"),
        }

    @staticmethod
    def _compute_metrics(eval_pred: Any) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }



def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _filter_labeled(
    clauses: list[str],
    labels: list[str],
) -> tuple[list[str], list[str]]:
    """Remove 'Unknown' labelled examples for supervised training."""
    filtered_c, filtered_l = zip(
        *[(c, l) for c, l in zip(clauses, labels) if l != "Unknown"],
        strict=False,
    ) if any(l != "Unknown" for l in labels) else ([], [])
    return list(filtered_c), list(filtered_l)



def main(output_dir: str | None = None, clauses_file: str | None = None) -> None:
    _clauses_file = Path(clauses_file) if clauses_file else PROJECT_ROOT / "output" / "clauses.txt"
    _out = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    _out.mkdir(parents=True, exist_ok=True)
    intent_output = _out / "intent_classification.txt"

    if not _clauses_file.exists():
        logger.error("clauses.txt not found at %s — run Assignment 1 first.", _clauses_file)
        return

    clauses: list[str] = [
        line.strip()
        for line in _clauses_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d clauses from %s", len(clauses), _clauses_file)

    
    weak_labels = [weak_label(c) for c in clauses]
    labeled_clauses, labeled_labels = _filter_labeled(clauses, weak_labels)
    logger.info(
        "Weak labels: %d/%d clauses received a label.",
        len(labeled_clauses),
        len(clauses),
    )

    
    tfidf_preds: list[str] = []
    tfidf_clf = TFIDFIntentClassifier()
    if len(labeled_clauses) >= 8:
        tfidf_clf.fit(labeled_clauses, labeled_labels)
        tfidf_preds = tfidf_clf.predict(clauses)
        (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
        tfidf_clf.save(PROJECT_ROOT / "models" / "tfidf_intent_classifier.pkl")
    else:
        logger.warning(
            "Insufficient labeled data for TF-IDF training (%d examples). "
            "Using weak labels as TF-IDF predictions.",
            len(labeled_clauses),
        )
        tfidf_preds = weak_labels

    
    bert_results: list[tuple[str, float]] = []
    bert_clf: BERTIntentClassifier | None = None

    try:
        bert_clf = BERTIntentClassifier()
        if len(labeled_clauses) >= 8:
            bert_clf.train((labeled_clauses, labeled_labels), save_path="models/intent_bert")
        else:
            logger.warning(
                "Insufficient labeled data for BERT training (%d examples). "
                "Running zero-shot prediction only.",
                len(labeled_clauses),
            )
        bert_results = bert_clf.predict_with_confidence(clauses)
    except Exception as exc:
        logger.warning("BERT training/inference failed (%s); falling back to TF-IDF labels.", exc)
        bert_results = [(pred, 1.0) for pred in tfidf_preds]

    
    bert_labels = [label for label, _ in bert_results]
    bert_confidences = [conf for _, conf in bert_results]

    if len(labeled_clauses) >= 8:
        tfidf_metrics = tfidf_clf.evaluate(labeled_clauses, labeled_labels)
        logger.info("TF-IDF eval: accuracy=%.4f, F1_macro=%.4f",
                    tfidf_metrics["accuracy"], tfidf_metrics["f1_macro"])

        if bert_clf is not None:
            bert_metrics = bert_clf.evaluate((labeled_clauses, labeled_labels))
            logger.info("BERT eval:   accuracy=%.4f, F1_macro=%.4f",
                        bert_metrics["accuracy"], bert_metrics["f1_macro"])

    lines: list[str] = ["clause_text\ttfidf_label\tbert_label\tconfidence"]
    for clause, tfidf_pred, bert_label, conf in zip(
        clauses, tfidf_preds, bert_labels, bert_confidences
    ):
        safe_clause = clause.replace("\t", " ")
        lines.append(f"{safe_clause}\t{tfidf_pred}\t{bert_label}\t{conf:.4f}")

    intent_output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Intent classification written to %s (%d clauses).", intent_output, len(clauses))


if __name__ == "__main__":
    main()
