from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "output"
CLAUSES_FILE = OUTPUT_DIR / "clauses.txt"


def _check_prerequisites() -> bool:

    if not CLAUSES_FILE.exists():
        logger.error(
            "Prerequisite missing: %s\n"
            "Please run Assignment 1 first (python src/assignment1/run_assignment1.py).",
            CLAUSES_FILE,
        )
        return False
    lines = [
        ln for ln in CLAUSES_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if not lines:
        logger.error("%s is empty — run Assignment 1 first.", CLAUSES_FILE)
        return False
    logger.info("Found %d clauses in %s", len(lines), CLAUSES_FILE)
    return True


def _run_step(name: str, step_fn: Any) -> bool:

    logger.info("=" * 60)
    logger.info("Starting: %s", name)
    start = time.perf_counter()
    try:
        step_fn()
        logger.info("Completed: %s  [%.1fs]", name, time.perf_counter() - start)
        return True
    except Exception as exc:
        logger.error(
            "FAILED: %s  [%.1fs]  — %s: %s",
            name,
            time.perf_counter() - start,
            type(exc).__name__,
            exc,
        )
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _check_prerequisites():
        sys.exit(1)

    overall_start = time.perf_counter()
    results: dict[str, bool] = {}

    def run_ner() -> None:
        from assignment2.ner_model import main as ner_main

        ner_main()

    results["Task 2.1 — NER"] = _run_step(
        "Task 2.1 — Named Entity Recognition", run_ner
    )

    def run_srl() -> None:
        from assignment2.srl_model import main as srl_main

        srl_main()

    results["Task 2.2 — SRL"] = _run_step("Task 2.2 — Semantic Role Labeling", run_srl)

    def run_intent() -> None:
        from assignment2.intent_classifier import main as intent_main

        intent_main()

    results["Task 2.3 — Intent"] = _run_step(
        "Task 2.3 — Intent Classification", run_intent
    )

    total_elapsed = time.perf_counter() - overall_start
    logger.info("=" * 60)
    logger.info("Assignment 2 Summary  [total: %.1fs]", total_elapsed)
    for task_name, success in results.items():
        logger.info("  %-35s %s", task_name, "OK" if success else "FAILED")

    logger.info("Output files in output/:")
    for fname in ("ner_results.json", "srl_results.json", "intent_classification.txt"):
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            logger.info("  %-45s %.1f KB", fname, fpath.stat().st_size / 1024)
        else:
            logger.info("  %-45s (not created)", fname)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
