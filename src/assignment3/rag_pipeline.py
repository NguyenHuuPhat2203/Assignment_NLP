"""
RAG Pipeline — LangChain 0.2 LCEL + Gemini for legal contract Q&A.

Uses LangChain Expression Language (LCEL), NOT the deprecated LLMChain.
Retrieves relevant clauses from ChromaDB, then generates answers with Gemini.
Anti-hallucination: LLM is constrained to answer ONLY from retrieved context.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .vector_store import LegalVectorStore

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a legal contract analyst assistant. Your role is to answer questions about "
    "legal contracts based ONLY on the provided contract clauses.\n\n"
    "Rules:\n"
    "1. Answer ONLY using information from the provided contract clauses\n"
    "2. Always cite the clause number (e.g., [Clause 3]) when referencing specific text\n"
    "3. If the answer cannot be found in the provided clauses, respond with:\n"
    '   "This information is not found in the provided contract clauses."\n'
    "4. Do NOT add information, interpretations, or assumptions not present in the context\n"
    "5. Be concise and precise in your answers\n\n"
    "Contract clauses:\n"
    "{context}"
)


class LegalRAGPipeline:
    """LangChain LCEL RAG pipeline for legal contract question answering."""

    def __init__(
        self,
        vector_store: LegalVectorStore,
        model_name: str | None = None,
        top_k: int = 3,
    ) -> None:
        """Initialise the pipeline with a vector store and a Gemini LLM.

        Args:
            vector_store: Initialised :class:`LegalVectorStore` instance.
            model_name: Gemini model identifier. Falls back to ``GEMINI_MODEL``
                env var, then ``gemini-1.5-flash``.
            top_k: Number of clauses to retrieve per query.
        """
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Copy .env.example to .env and fill in your API key."
            )

        resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=0.1,
            google_api_key=api_key,
        )
        self._vector_store = vector_store
        self._top_k = top_k
        self._chain = self._build_chain()
        logger.info("LegalRAGPipeline ready (model=%s, top_k=%d).", resolved_model, top_k)

    

    def _format_context(self, results: list[dict]) -> str:
        """Format retrieved clauses as a numbered context block.

        Args:
            results: Output of :meth:`LegalVectorStore.search`.

        Returns:
            Multi-line string with each clause on its own numbered line.
        """
        if not results:
            return "No relevant clauses found."

        lines: list[str] = []
        for rank, r in enumerate(results, start=1):
            clause_num = r["clause_id"] + 1  # 1-based for human readability
            lines.append(f"[Clause {clause_num}] (relevance: {r['score']:.3f})\n{r['text']}")
        return "\n\n".join(lines)

    def _build_chain(self):
        """Build and return the LCEL chain: prompt | LLM | output_parser.

        Returns:
            A LangChain Runnable that accepts ``{"context": str, "question": str}``.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    

    def query(self, question: str) -> dict[str, Any]:
        """Answer a single question using RAG.

        Retrieves relevant clauses, injects them as context, and invokes Gemini.

        Args:
            question: Natural-language question about the contract.

        Returns:
            Dict with keys:
            - ``answer`` (str): Generated answer.
            - ``sources`` (list[dict]): Retrieved clause dicts from the vector store.
            - ``question`` (str): Original question echoed back.
        """
        sources = self._vector_store.search(question, top_k=self._top_k)
        context = self._format_context(sources)

        try:
            answer: str = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        return {"answer": answer, "sources": sources, "question": question}

    def chat(self, messages: list[dict]) -> dict[str, Any]:
        """Answer the latest message in a multi-turn conversation.

        Earlier turns are prepended as ``HumanMessage``/``AIMessage`` pairs so
        the model retains context. The last message must be a user turn.

        Args:
            messages: List of ``{"role": "user"|"assistant", "content": str}``
                dicts representing the conversation history.

        Returns:
            Dict with keys ``answer``, ``sources``, ``question`` (same as
            :meth:`query`).
        """
        if not messages:
            return {"answer": "", "sources": [], "question": ""}

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        sources = self._vector_store.search(last_user_msg, top_k=self._top_k)
        context = self._format_context(sources)

        # Build history list for the prompt (excluding the last user message)
        history_messages: list = [
            SystemMessage(content=_SYSTEM_PROMPT.format(context=context))
        ]
        history = messages[:-1]
        for msg in history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))

        history_messages.append(HumanMessage(content=last_user_msg))

        try:
            response = self._llm.invoke(history_messages)
            answer: str = StrOutputParser().invoke(response)
        except Exception as exc:
            logger.error("LLM chat invocation failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        return {"answer": answer, "sources": sources, "question": last_user_msg}
