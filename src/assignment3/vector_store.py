from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LegalVectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "legal_clauses",
    ) -> None:

        self._persist_dir = persist_dir
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "LegalVectorStore ready — collection '%s', size=%d",
            collection_name,
            self.get_collection_size(),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:

        vectors = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return vectors.tolist()

    def load_clauses(self, clauses_file: str = "output/clauses.txt") -> list[str]:

        path = Path(clauses_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Clauses file not found: {path}. Run Assignment 1 first."
            )
        clauses = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ]
        return [c for c in clauses if c]

    def load_metadata(
        self,
        ner_file: str = "output/ner_results.json",
        srl_file: str = "output/srl_results.json",
        intent_file: str = "output/intent_classification.txt",
    ) -> list[dict]:

        ner_data: list = []
        srl_data: list = []

        ner_path = Path(ner_file)
        if ner_path.exists():
            try:
                ner_data = json.loads(ner_path.read_text(encoding="utf-8"))
                if not isinstance(ner_data, list):
                    ner_data = []
            except json.JSONDecodeError:
                logger.warning("Could not parse NER results from %s", ner_path)

        srl_path = Path(srl_file)
        if srl_path.exists():
            try:
                srl_data = json.loads(srl_path.read_text(encoding="utf-8"))
                if not isinstance(srl_data, list):
                    srl_data = []
            except json.JSONDecodeError:
                logger.warning("Could not parse SRL results from %s", srl_path)

        intent_pairs: list[tuple[str, str]] = []
        intent_path = Path(intent_file)
        if intent_path.exists():
            for line in intent_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("clause_text"):
                    continue
                parts = line.split("\t")
                tfidf_lbl = parts[1].strip() if len(parts) > 1 else ""
                bert_lbl = parts[2].strip() if len(parts) > 2 else tfidf_lbl
                intent_pairs.append((tfidf_lbl, bert_lbl))

        max_len = max(len(ner_data), len(srl_data), len(intent_pairs), 1)

        metadata_list: list[dict] = []
        for i in range(max_len):
            ner_entry = ner_data[i] if i < len(ner_data) else {}
            entities = (
                ner_entry.get("entities", []) if isinstance(ner_entry, dict) else []
            )
            ner_entities_str = json.dumps(entities, ensure_ascii=False)

            srl_entry = srl_data[i] if i < len(srl_data) else {}
            if isinstance(srl_entry, dict):
                predicate = str(srl_entry.get("predicate", ""))
                roles = srl_entry.get("roles", {})
                srl_roles_str = json.dumps(roles, ensure_ascii=False)
            else:
                predicate = ""
                srl_roles_str = "{}"

            intent_tfidf, intent_bert = (
                intent_pairs[i] if i < len(intent_pairs) else ("", "")
            )

            metadata_list.append(
                {
                    "ner_entities": ner_entities_str,
                    "srl_predicate": predicate,
                    "srl_roles": srl_roles_str,
                    "intent_tfidf": intent_tfidf,
                    "intent_bert": intent_bert,
                }
            )

        return metadata_list

    def index_clauses(
        self,
        clauses_file: str = "output/clauses.txt",
        metadata_files: Optional[dict] = None,
    ) -> int:

        if self.get_collection_size() > 0:
            logger.info(
                "Collection already indexed (%d docs). Skipping.",
                self.get_collection_size(),
            )
            return self.get_collection_size()

        clauses = self.load_clauses(clauses_file)
        if not clauses:
            logger.warning("No clauses found in %s", clauses_file)
            return 0

        mf = metadata_files or {}
        try:
            all_metadata = self.load_metadata(
                ner_file=mf.get("ner_file", "output/ner_results.json"),
                srl_file=mf.get("srl_file", "output/srl_results.json"),
                intent_file=mf.get("intent_file", "output/intent_classification.txt"),
            )
        except Exception as exc:
            logger.warning(
                "Could not load metadata (%s). Indexing without metadata.", exc
            )
            all_metadata = []

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for idx, clause in enumerate(clauses):
            ids.append(f"clause_{idx}")
            documents.append(clause)

            meta: dict = all_metadata[idx] if idx < len(all_metadata) else {}

            metadatas.append(
                {
                    "clause_index": str(idx),
                    "ner_entities": str(meta.get("ner_entities", "[]")),
                    "srl_predicate": str(meta.get("srl_predicate", "")),
                    "srl_roles": str(meta.get("srl_roles", "{}")),
                    "intent_tfidf": str(meta.get("intent_tfidf", "")),
                    "intent_bert": str(meta.get("intent_bert", "")),
                }
            )

        logger.info("Generating embeddings for %d clauses…", len(clauses))
        embeddings = self._embed(clauses)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        count = self.get_collection_size()
        logger.info(
            "Indexed %d clauses into collection '%s'.", count, self._collection_name
        )
        return count

    def search(self, query: str, top_k: int = 3) -> list[dict]:

        if self.get_collection_size() == 0:
            return []

        query_embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.get_collection_size()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            clause_id = int(meta.get("clause_index", -1))

            score = float(1.0 - dist)
            output.append(
                {
                    "clause_id": clause_id,
                    "text": doc,
                    "score": round(score, 4),
                    "metadata": meta,
                }
            )

        return output

    def get_collection_size(self) -> int:

        return self._collection.count()

    def clear(self) -> None:

        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared.", self._collection_name)
