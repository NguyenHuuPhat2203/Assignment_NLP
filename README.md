# Legal Contract Information Extraction and Semantic Analysis

> **NLP Course — HCMUT Semester 252 (2025-2026)**  
> A full NLP pipeline for English legal contract analysis, covering syntax, semantics, and a retrieval-augmented generation (RAG) chatbot.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Windows Setup](#windows-setup)
- [Assignment 1 — Preprocessing & Syntax Analysis](#assignment-1--preprocessing--syntax-analysis)
- [Assignment 2 — Information Extraction & Semantic Analysis](#assignment-2--information-extraction--semantic-analysis)
- [Assignment 3 — RAG Chatbot (Bonus)](#assignment-3--rag-chatbot-bonus)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Technical Decisions](#technical-decisions)
- [Dependencies](#dependencies)

---

## Overview

This project builds a three-stage NLP pipeline for analyzing English legal contracts using the [Atticus Open Contract Dataset](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta).

| Assignment | Topic | Output |
|---|---|---|
| **1** | Preprocessing & Syntax Analysis | `clauses.txt`, `chunks.txt`, `dependency.json` |
| **2** | Information Extraction & Semantics | `ner_results.json`, `srl_results.json`, `intent_classification.txt` |
| **3** *(Bonus)* | RAG Chatbot | Streamlit web app on `localhost:8501` |

**Stack**: Python 3.12 · spaCy 3.7 · HuggingFace Transformers · ChromaDB · LangChain 0.2 · Google Gemini · Streamlit

---

## Architecture

```
input/raw_contracts.txt
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  ASSIGNMENT 1 — Preprocessing & Syntax Analysis      │
│                                                      │
│  1.1 Clause Splitting                                │
│      spaCy dep-tree + rule-based SCONJ/CC splitting  │
│      → output/clauses.txt                            │
│                                                      │
│  1.2 Noun Phrase Chunking (IOB)                      │
│      spaCy noun_chunks → B-NP / I-NP / O tags        │
│      → output/chunks.txt                             │
│                                                      │
│  1.3 Dependency Parsing                              │
│      spaCy pre-trained parser (en_core_web_sm)       │
│      → output/dependency.json                        │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  ASSIGNMENT 2 — Information Extraction & Semantics   │
│                                                      │
│  2.1 Custom NER                                      │
│      Legal-BERT fine-tuned for domain entities       │
│      Entities: PARTY · MONEY · DATE · RATE ·         │
│                PENALTY · LAW                         │
│      → output/ner_results.json                       │
│                                                      │
│  2.2 Semantic Role Labeling (SRL)                    │
│      BERT + dependency-tree role mapping             │
│      Roles: Agent · Predicate · Theme ·              │
│             Recipient · Time · Condition             │
│      → output/srl_results.json                       │
│                                                      │
│  2.3 Clause Intent Classification                    │
│      Baseline: TF-IDF + Logistic Regression          │
│      Advanced: fine-tune bert-base-uncased           │
│      Labels: Obligation · Prohibition · Right ·      │
│              Termination Condition                   │
│      → output/intent_classification.txt              │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  ASSIGNMENT 3 — RAG Chatbot (Bonus)                  │
│                                                      │
│  Vector Store: ChromaDB + MiniLM embeddings          │
│  LLM: Google Gemini (gemini-1.5-flash)               │
│  Framework: LangChain 0.2 (LCEL)                     │
│  Interface: Streamlit web app                        │
│  Features: top-k retrieval · citations ·             │
│            anti-hallucination constraint             │
└──────────────────────────────────────────────────────┘
```

### Task Dependency Graph

```
input/raw_contracts.txt
         │
         ▼
    clause_splitter.py (1.1)
    output/clauses.txt
         │
    ┌────┼──────────────┬──────────────┬──────────────┐
    ▼    ▼              ▼              ▼              ▼
np_chunker dependency  ner_model    srl_model  intent_classifier
(1.2)    _parser(1.3) (2.1)        (2.2)      (2.3)
    │         │         │              │           │
    ▼         ▼         └──────────────┴───────────┘
chunks.txt dep.json                   │
                               vector_store.py (3)
                               rag_pipeline.py (3)
                               app.py (3)
```

---

## Project Structure

```
Assignment_NLP/
├── input/
│   └── raw_contracts.txt            # English legal contract text (your input)
├── output/                          # All generated outputs (created at runtime)
│   ├── clauses.txt                  # Task 1.1 — split clauses
│   ├── chunks.txt                   # Task 1.2 — IOB NP chunks
│   ├── dependency.json              # Task 1.3 — dependency parse trees
│   ├── ner_results.json             # Task 2.1 — named entities
│   ├── srl_results.json             # Task 2.2 — semantic role labels
│   └── intent_classification.txt   # Task 2.3 — intent labels
├── src/
│   ├── assignment1/
│   │   ├── clause_splitter.py       # Task 1.1
│   │   ├── np_chunker.py            # Task 1.2
│   │   ├── dependency_parser.py     # Task 1.3
│   │   └── run_assignment1.py       # Entry point
│   ├── assignment2/
│   │   ├── ner_model.py             # Task 2.1 — Legal-BERT NER
│   │   ├── srl_model.py             # Task 2.2 — SRL
│   │   ├── intent_classifier.py     # Task 2.3 — TF-IDF + BERT
│   │   └── run_assignment2.py       # Entry point
│   └── assignment3/
│       ├── vector_store.py          # ChromaDB wrapper
│       ├── rag_pipeline.py          # LangChain + Gemini RAG
│       └── app.py                   # Streamlit UI
├── data/
│   └── english/
│       ├── raw/                     # Atticus dataset raw files
│       └── sample_ner_training.json # NER training examples
├── models/                          # Saved fine-tuned model weights
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_assignment1.ipynb
│   ├── 03_assignment2.ipynb
│   └── 04_assignment3_rag.ipynb
├── scripts/
│   ├── setup_env.sh                 # Create venv + pip install
│   ├── download_datasets.sh         # Download via Kaggle API
│   └── run_all.sh                   # End-to-end pipeline runner
├── report/                          # LaTeX report source
├── .env.example                     # API key template
├── requirements.txt
└── IMPLEMENTATION_PLAN.md
```

---

## Quick Start

### 1. Prerequisites

| Requirement | Linux / macOS | Windows |
|---|---|---|
| **Python 3.12+** | `sudo apt install python3.12 python3.12-venv` or [python.org](https://www.python.org/downloads/) | [python.org/downloads](https://www.python.org/downloads/) — check **Add to PATH** |
| **Git** | Pre-installed or `sudo apt install git` | [git-scm.com/download/win](https://git-scm.com/download/win) |
| **Kaggle API key** | `~/.kaggle/kaggle.json` | `%USERPROFILE%\.kaggle\kaggle.json` |
| **Google API key** | [aistudio.google.com](https://aistudio.google.com/app/apikey) — Assignment 3 only | same |

---

### 2. Set Up the Environment

**Linux / macOS**
```bash
bash scripts/setup_env.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy RemoteSigned -File scripts\setup_env.ps1
```

> **First-time PowerShell users:** if you see an execution policy error, run this once in an admin PowerShell, then retry:
> ```powershell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Both scripts create `.venv/`, install all packages from `requirements.txt`, download spaCy models, and create a `.env` file from `.env.example`.

---

### 3. Configure API Keys

**Linux / macOS**
```bash
cp .env.example .env
nano .env          # set GOOGLE_API_KEY=your_key_here
```

**Windows (PowerShell)**
```powershell
copy .env.example .env
notepad .env       # set GOOGLE_API_KEY=your_key_here
```

---

### 4. Download the Dataset

**Linux / macOS**
```bash
bash scripts/download_datasets.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy RemoteSigned -File scripts\download_datasets.ps1
```

> Requires a valid Kaggle API key. Place `kaggle.json` at:
> - Linux/macOS: `~/.kaggle/kaggle.json`
> - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
>
> Get it at [kaggle.com/settings](https://www.kaggle.com/settings) → **Account** → **API** → **Create New Token**.

---

### 5. Prepare Input

Place your legal contract text in:

```
input/raw_contracts.txt
```

---

### 6. Run the Pipeline

#### Run each assignment individually

**Linux / macOS**
```bash
# Activate the environment
source .venv/bin/activate

# Assignment 1 — Syntax Analysis
python src/assignment1/run_assignment1.py

# Assignment 2 — Information Extraction (requires Assignment 1 output)
python src/assignment2/run_assignment2.py

# Assignment 3 — RAG Chatbot (requires GOOGLE_API_KEY in .env)
streamlit run src/assignment3/app.py
```

**Windows (PowerShell)**
```powershell
# Activate the environment
.venv\Scripts\Activate.ps1
# or without activating, use the venv Python directly:
# .venv\Scripts\python.exe src\assignment1\run_assignment1.py

# Assignment 1 — Syntax Analysis
python src\assignment1\run_assignment1.py

# Assignment 2 — Information Extraction (requires Assignment 1 output)
python src\assignment2\run_assignment2.py

# Assignment 3 — RAG Chatbot (requires GOOGLE_API_KEY in .env)
streamlit run src\assignment3\app.py
```

#### Or run Assignments 1 & 2 end-to-end

**Linux / macOS**
```bash
bash scripts/run_all.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy RemoteSigned -File scripts\run_all.ps1
```

---

## Windows Setup

> The project ships PowerShell scripts (`scripts\setup_env.ps1`, `scripts\download_datasets.ps1`, `scripts\run_all.ps1`) that mirror every bash script exactly. No WSL or manual steps required.

### Option A — Native Windows (PowerShell) Recommended

#### 1. Install Python 3.12+

Download from [python.org/downloads](https://www.python.org/downloads/). During installation:

- **Add Python to PATH**
- **Install pip**

Verify in a new terminal:
```powershell
python --version    # Python 3.12.x
```

#### 2. Install Git for Windows

Download from [git-scm.com](https://git-scm.com/download/win).

#### 3. Run the Setup Script

```powershell
cd C:\path\to\Assignment_NLP
powershell -ExecutionPolicy RemoteSigned -File scripts\setup_env.ps1
```

This creates `.venv\`, installs all packages (PyTorch CPU build via the official Windows wheel index, then everything else from `requirements.txt`), downloads spaCy models, and creates `.env`.

#### 4. Continue with Quick Start

Steps 3–6 in [Quick Start](#quick-start) each show a **Windows (PowerShell)** block — follow those.

---

### Option B — WSL 2

WSL 2 gives you a full Linux environment. The bash scripts run without modification.

```powershell
# Install WSL 2 (run as Administrator, then restart)
wsl --install
```

Then inside Ubuntu:
```bash
# Install Python 3.12
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Navigate to your project (Windows files are mounted under /mnt/c/)
cd /mnt/c/Users/<YourUsername>/Desktop/Assignment_NLP

# Run the bash setup script
bash scripts/setup_env.sh
```

All subsequent steps are identical to the Linux/macOS path in [Quick Start](#quick-start).

---

### Windows Troubleshooting

| Issue | Solution |
|---|---|
| `python` not found | Reinstall Python with **Add to PATH** checked, or use `py` instead of `python` |
| Script blocked by execution policy | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` in admin PowerShell |
| `pip install chromadb` fails | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| `torch` import error | Re-run `setup_env.ps1`; ensure the `--index-url https://download.pytorch.org/whl/cpu` wheel was used |
| CUDA GPU not detected | Replace CPU wheel URL with `cu121` or `cu118` in `setup_env.ps1` step 3 |
| `streamlit run` opens wrong browser | Open `http://localhost:8501` manually, or add `--server.headless true` |
| Long path errors | `git config --system core.longpaths true` and enable **Win32 Long Paths** in Group Policy |
| `UnicodeDecodeError` on contract file | Save `raw_contracts.txt` as **UTF-8** (Notepad → Save As → Encoding: UTF-8 without BOM) |

---

## Assignment 1 — Preprocessing & Syntax Analysis

### Task 1.1 — Clause Splitting (`src/assignment1/clause_splitter.py`)

Splits raw contract text into atomic clauses using a hybrid rule + dependency-tree approach.

**Algorithm:**
1. Parse the document with spaCy `en_core_web_sm`
2. Traverse each sentence's dependency tree
3. Split at coordinating conjunctions (CC) where both sides have a subject + verb
4. Split at subordinating conjunctions: `if`, `unless`, `provided that`, `whereas`, `subject to`, `upon`
5. Preserve conditional pairs (condition + consequence) intact

**Output** — `output/clauses.txt`: one clause per line.

---

### Task 1.2 — NP Chunking (`src/assignment1/np_chunker.py`)

Tags each token in every clause with IOB noun-phrase labels.

**Algorithm:**
1. Run spaCy `doc.noun_chunks` on each clause
2. Assign IOB tags: `B-NP` (first token of an NP), `I-NP` (continuation), `O` (outside)
3. Output format: `token\tIOB-tag` per line, blank line between clauses

**Output** — `output/chunks.txt`:
```
The     B-NP
Employer B-NP
shall   O
pay     O
...
```

---

### Task 1.3 — Dependency Parsing (`src/assignment1/dependency_parser.py`)

Extracts the full dependency parse tree for each clause using spaCy's pre-trained parser (no fine-tuning required).

**Per-token output fields:** `id`, `token`, `lemma`, `pos`, `head_id`, `head_token`, `dep_rel`

**Output** — `output/dependency.json`: a JSON array of clause parse objects.

---

## Assignment 2 — Information Extraction & Semantic Analysis

> **Prerequisite:** Assignment 1 must be run first to produce `output/clauses.txt`.

### Task 2.1 — Custom NER (`src/assignment2/ner_model.py`)

Fine-tunes `nlpaueb/legal-bert-base-uncased` for token classification on six legal entity types.

| Label | Examples |
|---|---|
| `PARTY` | Party A, Employer, Lessee |
| `MONEY` | $5,000, USD 1,000,000 |
| `DATE` | 05 May 2024, within 30 days |
| `RATE` | 1% per day, 12% per annum |
| `PENALTY` | late fee, liquidated damages |
| `LAW` | Civil Code Article 10, Law No. 91/2015/QH13 |

**Fallback:** Rule-based `EntityRuler` with regex patterns for `MONEY`, `DATE`, and `RATE` when no training data is available.

**Evaluation:** seqeval F1 on a held-out 20% split.

**Output** — `output/ner_results.json`

---

### Task 2.2 — Semantic Role Labeling (`src/assignment2/srl_model.py`)

Maps dependency relations to semantic roles using a rule-based approach on top of spaCy's dependency tree.

| Semantic Role | Dependency Relation(s) |
|---|---|
| Agent | `nsubj`, `nsubjpass` |
| Predicate | `ROOT` verb |
| Theme | `dobj`, `obj`, `nsubjpass` |
| Recipient | `iobj`, prep `to` |
| Time | prep `before/after/on/within`, `ARGM-TMP` |
| Condition | `advcl` (if/unless), `mark` |

> AllenNLP's SRL model has Python 3.12 compatibility issues; the dependency-tree mapping is used as a robust alternative.

**Output** — `output/srl_results.json`

---

### Task 2.3 — Intent Classification (`src/assignment2/intent_classifier.py`)

Classifies each clause into one of four legal intents using two models for comparison.

**Weak labeling heuristics (for training data generation):**

| Intent | Keywords / Patterns |
|---|---|
| Obligation | `shall`, `must`, `required to`, `is obligated` |
| Prohibition | `shall not`, `must not`, `prohibited`, `is forbidden` |
| Right | `may`, `is entitled to`, `has the right`, `can` |
| Termination Condition | `terminate`, `termination`, `expiry`, `upon breach` |

**Models:**
- **Baseline:** TF-IDF (1–3 ngrams) + Logistic Regression (`C=1.0`)
- **Advanced:** Fine-tuned `bert-base-uncased` (4 classes, 3 epochs)

**Output** — `output/intent_classification.txt`: one `clause_id\tintent` entry per line.

---

## Assignment 3 — RAG Chatbot (Bonus)

> **Prerequisite:** Assignments 1 and 2 must be run first. Requires `GOOGLE_API_KEY` in `.env`.

A Streamlit-based Q&A application that answers questions about the contract strictly from the indexed clauses — no hallucination.

### Indexing Pipeline

1. Load `output/clauses.txt` (each clause = one document)
2. Attach NER, SRL, and intent metadata from Assignment 2 outputs
3. Embed with `sentence-transformers/all-MiniLM-L6-v2`
4. Store in ChromaDB (local persistent collection at `./chroma_db/`)

### Query Pipeline

1. Embed the user question
2. Retrieve the top-k most similar clauses by cosine similarity (default `k=3`)
3. Format the retrieved context with clause numbers
4. Generate an answer with Gemini constrained to the context only
5. Return answer + source clause citations

### Anti-Hallucination System Prompt

```
You are a legal contract analyst assistant. Answer ONLY based on the contract
clauses provided below. Always cite the clause number (e.g., [Clause 3]).
If the answer cannot be found in the provided clauses, state:
"This information is not found in the contract."
Do NOT add information not present in the provided context.
```

### App Features

- **🔨 Build Index** — indexes all clauses on first run
- **Chat interface** — persistent message history across the session
- **📎 Sources** — expandable panel showing retrieved clauses with similarity scores
- **Intent badges** — color-coded: Obligation (orange), Prohibition (red), Right (green), Termination Condition (purple)
- **Sidebar controls** — top-k slider (1–10), metadata toggle (NER entities + SRL roles)

### Launch

```bash
streamlit run src/assignment3/app.py
```

App will be available at `http://localhost:8501`.

---

## Configuration

Copy `.env.example` to `.env` and set your values:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RESULTS=3
```

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio key (required for Assignment 3) | — |
| `GEMINI_MODEL` | Gemini model to use | `gemini-1.5-flash` |
| `CHROMA_PERSIST_DIR` | Path for ChromaDB persistence | `./chroma_db` |
| `TOP_K_RESULTS` | Default number of retrieved clauses | `3` |

---

## Output Files

| File | Produced by | Format |
|---|---|---|
| `output/clauses.txt` | Task 1.1 | Plain text, one clause per line |
| `output/chunks.txt` | Task 1.2 | TSV `token\tIOB-tag`, blank line between clauses |
| `output/dependency.json` | Task 1.3 | JSON array of per-clause token objects |
| `output/ner_results.json` | Task 2.1 | JSON array of clause NER results |
| `output/srl_results.json` | Task 2.2 | JSON array of clause SRL results |
| `output/intent_classification.txt` | Task 2.3 | TSV `clause_id\tintent_label` |

---

## Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | English | Atticus dataset is well-structured for legal NER/classification |
| Clause splitting | spaCy dep tree + rules | Deterministic, explainable, no training required |
| NP chunking | `spaCy.noun_chunks` | Built-in, accurate; IOB conversion is trivial |
| Dependency parsing | `en_core_web_sm` | Pre-trained, fast; spec does not require training |
| NER model | Legal-BERT fine-tune | Domain-specific; outperforms general BERT on legal text |
| SRL approach | Dep-tree rule mapping | AllenNLP has Python 3.12 compat issues; dep-tree is robust |
| Intent baseline | TF-IDF + LogReg | Spec requires comparison; fast and interpretable |
| Intent advanced | `bert-base-uncased` fine-tune | Best accuracy for 4-class intent classification |
| RAG LLM | Gemini 1.5-flash | Already in requirements; free tier available |
| Vector DB | ChromaDB | Already in requirements; local persistent; easy Python API |
| RAG framework | LangChain 0.2 LCEL | Modern pattern; already in requirements |
| UI | Streamlit | Fastest to build; already in requirements |

---

## Dependencies

All versions are expressed as minimums (`>=`) in `requirements.txt` so pip freely resolves the latest compatible release for Python 3.12+. No strict pins.

| Category | Library | Minimum |
|---|---|---|
| Core NLP | `spacy`, `spacy-transformers` | 3.7, 1.3 |
| Vietnamese NLP | `underthesea` | 6.8 |
| Transformers | `transformers`, `tokenizers`, `accelerate` | 4.44, 0.19, 0.30 |
| Deep Learning | `torch`, `torchvision`, `torchaudio` | 2.2, 0.17, 2.2 |
| ML / Sklearn | `scikit-learn`, `numpy`, `pandas` | 1.3, 1.26, 2.1 |
| HuggingFace | `datasets`, `evaluate`, `seqeval` | 2.16, 0.4, 1.2.2 |
| Embeddings | `sentence-transformers` | 3.0 |
| Vector DB | `chromadb` | 0.4 |
| LangChain | `langchain`, `langchain-community`, `langchain-google-genai` | 0.2, 0.2, 1.0 |
| LLM | `google-generativeai` | 0.7 |
| UI | `streamlit`, `fastapi`, `uvicorn` | 1.30, 0.100, 0.30 |
| Utilities | `python-dotenv`, `tqdm`, `kaggle`, `openpyxl` | 1.0, 4.66, 1.6, 3.1 |

**Linux / macOS** — install all at once:
```bash
pip install -r requirements.txt
```

**Windows** — PyTorch needs the official wheel index first:
```powershell
# Install PyTorch (CPU) via Windows wheel index
pip install "torch>=2.2" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install everything else
pip install -r requirements.txt
```

Download the spaCy English model (required for Assignment 1):
```bash
python -m spacy download en_core_web_sm
```
