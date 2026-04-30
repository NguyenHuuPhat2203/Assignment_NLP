"""
Assignment 1 Runner

Executes all Assignment 1 tasks in order and writes outputs to output/ as required
by the assignment specification:
    output/clauses.txt
    output/chunks.txt
    output/dependency.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assignment1.clause_splitter import main as run_clause_splitter
from assignment1.dependency_parser import main as run_dependency_parser
from assignment1.np_chunker import main as run_np_chunker


def main() -> None:
    (PROJECT_ROOT / "output").mkdir(parents=True, exist_ok=True)

    tasks = [
        ("Task 1.1 — Clause Splitting", run_clause_splitter),
        ("Task 1.2 — NP Chunking", run_np_chunker),
        ("Task 1.3 — Dependency Parsing", run_dependency_parser),
    ]

    for name, fn in tasks:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            fn()
            print(f"  ✓ Completed in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            raise

    output_dir = PROJECT_ROOT / "output"
    print("\n✅ Assignment 1 complete.")
    for fname in ("clauses.txt", "chunks.txt", "dependency.json"):
        fpath = output_dir / fname
        if fpath.exists():
            print(f"   {fname:30s}  {fpath.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
