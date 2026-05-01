from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    domain: str
    title: str
    url: str
    chunk_text: str
    score: float

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "title": self.title,
            "url": self.url,
            "chunk_text": self.chunk_text,
        }


class SupportRetriever:
    CHUNK_WORDS = 220
    CHUNK_OVERLAP = 40

    def __init__(
        self,
        corpus_dir: str | Path = "corpus",
        index_path: str | Path = "retriever.index",
        meta_path: str | Path = "retriever_meta.pkl",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name
        self._model = None
        self._model_load_failed = False
        self._faiss = None
        self.index = None
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.available_domains: set[str] = set()
        self._signature = self._compute_corpus_signature()
        self._load_or_build_index()

    def _lazy_import_model(self):
        if self._model_load_failed:
            raise RuntimeError("Embedding model is unavailable for this run.")
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception as local_error:
                self._model_load_failed = True
                raise RuntimeError(
                    f"Embedding model cache was unavailable for {self.model_name}: {local_error}"
                ) from local_error
        return self._model

    def _lazy_import_faiss(self):
        if self._faiss is None:
            import faiss

            self._faiss = faiss
        return self._faiss

    def _compute_corpus_signature(self) -> list[tuple[str, int, int]]:
        signature: list[tuple[str, int, int]] = []
        for path in sorted(self.corpus_dir.glob("*.json")):
            stat = path.stat()
            signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        return signature

    def _load_corpus_articles(self) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        if not self.corpus_dir.exists():
            LOGGER.warning("Corpus directory %s does not exist.", self.corpus_dir)
            return articles

        for path in sorted(self.corpus_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.warning("Failed to read corpus file %s: %s", path, exc)
                continue

            if not isinstance(data, list):
                LOGGER.warning("Corpus file %s did not contain a list.", path)
                continue

            articles.extend(item for item in data if isinstance(item, dict))

        return articles

    def _chunk_article(self, article: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(article.get("content", "")).strip()
        if not text:
            return []

        words = text.split()
        chunks: list[dict[str, Any]] = []
        step = self.CHUNK_WORDS - self.CHUNK_OVERLAP
        for start in range(0, len(words), step):
            end = min(len(words), start + self.CHUNK_WORDS)
            chunk_words = words[start:end]
            if len(chunk_words) < 40:
                continue
            chunks.append(
                {
                    "domain": article.get("domain", "Unknown"),
                    "title": article.get("title", "Untitled"),
                    "url": article.get("url", ""),
                    "chunk_text": " ".join(chunk_words),
                }
            )
            if end >= len(words):
                break
        return chunks

    def _build_index(self) -> None:
        articles = self._load_corpus_articles()
        self.chunks = []
        for article in articles:
            self.chunks.extend(self._chunk_article(article))

        self.available_domains = {chunk["domain"] for chunk in self.chunks}
        if not self.chunks:
            LOGGER.warning("No corpus chunks were available. Retrieval will be empty.")
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.index = None
            self._save_cache()
            return

        try:
            model = self._lazy_import_model()
            faiss = self._lazy_import_faiss()
            texts = [chunk["chunk_text"] for chunk in self.chunks]
            self.embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except Exception as exc:
            LOGGER.warning(
                "Embedding index build failed; using lexical retrieval only for this run: %s",
                exc,
            )
            self._model_load_failed = True
            self.embeddings = None
            self.index = None
        self._save_cache()

    def _save_cache(self) -> None:
        meta = {
            "signature": self._signature,
            "chunks": self.chunks,
            "available_domains": sorted(self.available_domains),
            "embeddings": self.embeddings,
        }
        with self.meta_path.open("wb") as handle:
            pickle.dump(meta, handle)

        if self.index is not None:
            faiss = self._lazy_import_faiss()
            faiss.write_index(self.index, str(self.index_path))

    def _load_cached_index(self) -> bool:
        if not self.meta_path.exists():
            return False

        try:
            with self.meta_path.open("rb") as handle:
                meta = pickle.load(handle)
        except Exception as exc:
            LOGGER.warning("Failed to load retrieval metadata cache: %s", exc)
            return False

        if meta.get("signature") != self._signature:
            return False

        self.chunks = meta.get("chunks", [])
        self.available_domains = set(meta.get("available_domains", []))
        self.embeddings = meta.get("embeddings")
        self.index = None

        if self.embeddings is None:
            self._model_load_failed = True
            return True

        if self.index_path.exists() and self.embeddings is not None:
            try:
                faiss = self._lazy_import_faiss()
                self.index = faiss.read_index(str(self.index_path))
            except Exception as exc:
                LOGGER.warning("Failed to load FAISS index cache: %s", exc)
                return False
        return True

    def _load_or_build_index(self) -> None:
        if self._load_cached_index():
            return
        self._build_index()

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not query.strip() or not self.chunks:
            return []
        if self._model_load_failed or self.embeddings is None:
            return self._lexical_retrieve(query=query, domain=domain, top_k=top_k)

        try:
            model = self._lazy_import_model()
            query_embedding = model.encode(
                [query],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].astype("float32")
        except Exception as exc:
            self._model_load_failed = True
            LOGGER.warning(
                "Falling back to lexical retrieval because query embedding failed: %s",
                exc,
            )
            return self._lexical_retrieve(query=query, domain=domain, top_k=top_k)

        if domain and domain != "None":
            candidate_ids = self._candidate_ids_for_domain(domain)
            if not candidate_ids:
                return []

            if self.embeddings is None:
                return self._lexical_retrieve(query=query, domain=domain, top_k=top_k)

            candidate_embeddings = self.embeddings[candidate_ids]
            scores = candidate_embeddings @ query_embedding
            ranked_local_ids = np.argsort(scores)[::-1][:top_k]
            results = []
            for local_id in ranked_local_ids:
                global_id = candidate_ids[int(local_id)]
                chunk = self.chunks[global_id]
                results.append(
                    RetrievedChunk(
                        domain=chunk["domain"],
                        title=chunk["title"],
                        url=chunk["url"],
                        chunk_text=chunk["chunk_text"],
                        score=float(scores[int(local_id)]),
                    )
                )
            return results

        if self.index is None:
            return self._lexical_retrieve(query=query, domain=domain, top_k=top_k)

        search_k = min(max(top_k, 1), len(self.chunks))
        scores, indices = self.index.search(np.array([query_embedding]), search_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    domain=chunk["domain"],
                    title=chunk["title"],
                    url=chunk["url"],
                    chunk_text=chunk["chunk_text"],
                    score=float(score),
                )
            )
        return results

    def has_domain_coverage(self, domain: str) -> bool:
        return domain.lower() in {available.lower() for available in self.available_domains}

    def _candidate_ids_for_domain(self, domain: str) -> list[int]:
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk["domain"].lower() == domain.lower()
        ]

    def _lexical_retrieve(
        self,
        query: str,
        domain: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        if domain and domain != "None":
            candidate_ids = self._candidate_ids_for_domain(domain)
        else:
            candidate_ids = list(range(len(self.chunks)))

        scored_candidates: list[tuple[float, int]] = []
        for idx in candidate_ids:
            chunk = self.chunks[idx]
            chunk_terms = self._tokenize(chunk["chunk_text"])
            overlap = query_terms.intersection(chunk_terms)
            if not overlap:
                continue

            score = float(len(overlap))
            if len(query_terms) > 0:
                score += len(overlap) / len(query_terms)
            score += self._metadata_score_bonus(chunk)
            scored_candidates.append((score, idx))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        for score, idx in scored_candidates[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    domain=chunk["domain"],
                    title=chunk["title"],
                    url=chunk["url"],
                    chunk_text=chunk["chunk_text"],
                    score=score,
                )
            )
        return results

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in "".join(
                char.lower() if char.isalnum() else " "
                for char in text
            ).split()
            if len(token) > 2
        }

    def _metadata_score_bonus(self, chunk: dict[str, Any]) -> float:
        title = str(chunk.get("title", "")).lower()
        url = str(chunk.get("url", "")).lower()
        bonus = 0.0

        preferred_markers = ["support", "consumer", "travel-support", "faq"]
        if any(marker in title or marker in url for marker in preferred_markers):
            bonus += 1.5

        discouraged_markers = ["run-your-business", "small-business", "merchant", "programme"]
        if any(marker in title or marker in url for marker in discouraged_markers):
            bonus -= 1.5

        return bonus
