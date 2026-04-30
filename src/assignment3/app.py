"""
Streamlit RAG Chatbot — Legal Contract Q&A

Features:
- Load and index contract clauses from Assignment 1 & 2 outputs
- Chat interface with message history
- Retrieved source clauses with metadata display
- Intent badges (color-coded)
- Configurable top-k retrieval
- Anti-hallucination constraints
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so relative output/ paths work regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st

from assignment3.rag_pipeline import LegalRAGPipeline
from assignment3.vector_store import LegalVectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENT_COLORS: dict[str, str] = {
    "Obligation": "#FF9900",
    "Prohibition": "#FF3333",
    "Right": "#00AA44",
    "Termination Condition": "#9933CC",
    "Unknown": "#999999",
}

_CLAUSES_FILE = str(_PROJECT_ROOT / "output" / "clauses.txt")
_NER_FILE = str(_PROJECT_ROOT / "output" / "ner_results.json")
_SRL_FILE = str(_PROJECT_ROOT / "output" / "srl_results.json")
_INTENT_FILE = str(_PROJECT_ROOT / "output" / "intent_classification.txt")
_CHROMA_DIR = str(_PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").lstrip("./"))

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    title="Legal Contract Q&A",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# API key guard
# ---------------------------------------------------------------------------

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "**GOOGLE_API_KEY is not set.**\n\n"
        "1. Copy `.env.example` to `.env` in the project root.\n"
        "2. Set `GOOGLE_API_KEY=<your_key>` in that file.\n"
        "3. Restart the app with `streamlit run src/assignment3/app.py`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = LegalVectorStore(persist_dir=_CHROMA_DIR)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = LegalRAGPipeline(
        vector_store=st.session_state.vector_store,
        top_k=int(os.getenv("TOP_K_RESULTS", "3")),
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ Legal Contract RAG")
    st.markdown(
        "Ask questions about your legal contract. "
        "The assistant retrieves relevant clauses and answers "
        "strictly from the contract text — no hallucination."
    )
    st.divider()

    top_k: int = st.slider("Top-K Results", min_value=1, max_value=10, value=3)
    st.session_state.pipeline._top_k = top_k

    st.divider()

    # --- Index status ---
    collection_size = st.session_state.vector_store.get_collection_size()
    if collection_size > 0:
        st.success(f"✅ Index ready — {collection_size} clauses indexed")
    else:
        st.warning("⚠️ Index is empty. Click **Build Index** to index clauses.")

    # --- Build Index button ---
    if st.button("🔨 Build Index", use_container_width=True):
        with st.spinner("Indexing clauses… this may take a moment."):
            try:
                count = st.session_state.vector_store.index_clauses(
                    clauses_file=_CLAUSES_FILE,
                    metadata_files={
                        "ner_file": _NER_FILE,
                        "srl_file": _SRL_FILE,
                        "intent_file": _INTENT_FILE,
                    },
                )
                st.success(f"✅ Indexed {count} clauses successfully!")
                st.rerun()
            except FileNotFoundError as exc:
                st.error(
                    f"**Clauses file not found.**\n\n{exc}\n\n"
                    "Run Assignment 1 first to generate `output/clauses.txt`."
                )
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    # --- Clear Index button ---
    if st.button("🗑️ Clear Index", use_container_width=True):
        st.session_state.vector_store.clear()
        st.success("Index cleared.")
        st.rerun()

    st.divider()
    show_metadata: bool = st.checkbox("Show Metadata", value=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent_badge(intent_label: str) -> str:
    """Return an HTML badge span for an intent label.

    Args:
        intent_label: Intent string (e.g. ``"Obligation"``).

    Returns:
        HTML string rendering a colour-coded badge.
    """
    color = INTENT_COLORS.get(intent_label, INTENT_COLORS["Unknown"])
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:2px 8px;border-radius:10px;font-size:0.8em;'
        f'font-weight:bold">{intent_label or "Unknown"}</span>'
    )


def _render_sources(sources: list[dict], show_meta: bool) -> None:
    """Render retrieved source clauses inside a Streamlit expander.

    Args:
        sources: List of result dicts from :meth:`LegalVectorStore.search`.
        show_meta: Whether to display NER and SRL metadata sections.
    """
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} clause{'s' if len(sources) != 1 else ''})"):
        for result in sources:
            clause_num = result["clause_id"] + 1
            meta = result.get("metadata", {})
            intent = meta.get("intent_bert") or meta.get("intent_tfidf") or "Unknown"

            st.markdown(f"**Clause {clause_num}** (score: {result['score']:.3f})")
            st.markdown(_intent_badge(intent), unsafe_allow_html=True)
            st.markdown(f"> {result['text']}")

            if show_meta:
                col_ner, col_srl = st.columns(2)
                with col_ner:
                    st.markdown("**NER Entities**")
                    try:
                        entities = json.loads(meta.get("ner_entities", "[]"))
                        if entities:
                            for ent in entities:
                                if isinstance(ent, dict):
                                    st.markdown(
                                        f"- `{ent.get('text', '')}` → *{ent.get('label', '')}*"
                                    )
                                else:
                                    st.markdown(f"- `{ent}`")
                        else:
                            st.caption("No entities")
                    except (json.JSONDecodeError, TypeError):
                        st.caption("No entities")

                with col_srl:
                    st.markdown("**SRL Roles**")
                    predicate = meta.get("srl_predicate", "")
                    if predicate:
                        st.markdown(f"Predicate: `{predicate}`")
                    try:
                        roles = json.loads(meta.get("srl_roles", "{}"))
                        if roles:
                            for role, value in roles.items():
                                st.markdown(f"- **{role}**: {value}")
                        else:
                            st.caption("No roles")
                    except (json.JSONDecodeError, TypeError):
                        st.caption("No roles")

            st.divider()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("⚖️ Legal Contract Q&A")
st.markdown("Ask any question about the legal contract. Answers are grounded in the contract text.")

# Display existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"], show_metadata)

# Accept new user input
if user_input := st.chat_input("Ask about the contract…"):
    # Append and render user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.vector_store.get_collection_size() == 0:
            answer = (
                "⚠️ The index is empty. "
                "Please click **🔨 Build Index** in the sidebar first."
            )
            sources: list[dict] = []
            st.markdown(answer)
        else:
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.pipeline.chat(
                        st.session_state.messages
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as exc:
                    answer = f"❌ Error: {exc}"
                    sources = []

            st.markdown(answer)
            _render_sources(sources, show_metadata)

    # Persist assistant message with sources for replay on re-render
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
