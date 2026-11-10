"""Chroma-compatible Collection backed by ZeroDB vectors."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

import requests


class Collection:
    """Chroma-compatible collection backed by ZeroDB vector storage.

    Supports ``add()``, ``query()``, ``get()``, ``update()``, ``delete()``,
    and ``count()`` with the same interface as chromadb.Collection.
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]],
        session: requests.Session,
        base_url: str,
        project_id: str,
    ):
        self.name = name
        self.metadata = metadata or {}
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._base_url}{path}"
        resp = self._session.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    def _vectors_path(self, suffix: str = "") -> str:
        """core#10222: project-scoped vectors path, replacing the old
        unscoped /api/v1/zerodb/vectors/* path this SDK originally called
        (which has no project-scoping concept -- it resolves only via the
        API key's owning user). `suffix` is appended as-is, e.g.
        "/upsert" or f"/{vector_id}"."""
        return f"/v1/projects/{self._project_id}/database/vectors{suffix}"

    def _prefixed_id(self, doc_id: str) -> str:
        """Prefix an ID with collection name for namespace isolation."""
        return f"{self.name}::{doc_id}"

    def _strip_prefix(self, prefixed_id: str) -> str:
        """Strip collection prefix from an ID."""
        prefix = f"{self.name}::"
        if prefixed_id.startswith(prefix):
            return prefixed_id[len(prefix):]
        return prefixed_id

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts via ZeroDB embedding API."""
        resp = self._request(
            "POST",
            "/api/v1/embeddings",
            json={"model": "bge-m3", "input": texts},
        )
        data = resp.json()
        # Handle OpenAI-compatible response format
        if isinstance(data, dict) and "data" in data:
            return [item["embedding"] for item in data["data"]]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected embedding response format: {type(data)}")

    def add(
        self,
        ids: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents and/or embeddings to the collection.

        At least one of ``documents`` or ``embeddings`` must be provided.
        If only documents are given, embeddings are auto-generated using
        free BAAI BGE models.

        Args:
            ids: Unique IDs for each item. Auto-generated if not provided.
            documents: Text documents to store and embed.
            embeddings: Pre-computed embedding vectors.
            metadatas: Per-document metadata dicts.
        """
        if documents is None and embeddings is None:
            raise ValueError("At least one of 'documents' or 'embeddings' must be provided.")

        n_items = len(documents) if documents else len(embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n_items)]
        if metadatas is None:
            metadatas = [{}] * n_items
        if documents is None:
            documents = [None] * n_items

        # Auto-embed if no embeddings provided
        if embeddings is None:
            texts_to_embed = [d for d in documents if d is not None]
            if texts_to_embed:
                embeddings = self._embed_texts(texts_to_embed)
            else:
                raise ValueError("No documents to embed and no embeddings provided.")

        for i in range(n_items):
            meta = dict(metadatas[i]) if metadatas[i] else {}

            self._request(
                "POST",
                self._vectors_path("/upsert"),
                json={
                    "vector_id": self._prefixed_id(ids[i]),
                    "vector_embedding": embeddings[i],
                    "vector_metadata": meta,
                    "document": documents[i] or "",
                    "namespace": self.name,
                },
            )

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List]:
        """Query the collection for similar documents.

        Args:
            query_texts: Text queries (auto-embedded).
            query_embeddings: Pre-computed query vectors.
            n_results: Number of results per query.
            where: Metadata filter dict.

        Returns:
            Dict with keys: ids, documents, metadatas, distances.
            Each value is a list of lists (one per query).
        """
        if query_texts is None and query_embeddings is None:
            raise ValueError("Provide query_texts or query_embeddings.")

        if query_embeddings is None:
            query_embeddings = self._embed_texts(query_texts)

        all_ids = []
        all_documents = []
        all_metadatas = []
        all_distances = []

        for embedding in query_embeddings:
            search_body: Dict[str, Any] = {
                "query_vector": embedding,
                "limit": n_results,
                "namespace": self.name,
            }
            if where:
                search_body["metadata_filter"] = dict(where)

            resp = self._request(
                "POST",
                self._vectors_path("/search"),
                json=search_body,
            )
            results = resp.json()

            # VectorSearchResult wraps matches; normalize defensively in
            # case the shape varies by server version.
            matches = results if isinstance(results, list) else results.get("results", results.get("matches", []))

            ids_batch = []
            docs_batch = []
            metas_batch = []
            dists_batch = []

            for match in matches:
                raw_id = match.get("vector_id", match.get("id", ""))
                ids_batch.append(self._strip_prefix(raw_id))
                meta = match.get("vector_metadata") or match.get("metadata") or {}
                docs_batch.append(match.get("document", ""))
                metas_batch.append(meta)
                dists_batch.append(match.get("distance", match.get("score", 0.0)))

            all_ids.append(ids_batch)
            all_documents.append(docs_batch)
            all_metadatas.append(metas_batch)
            all_distances.append(dists_batch)

        return {
            "ids": all_ids,
            "documents": all_documents,
            "metadatas": all_metadatas,
            "distances": all_distances,
        }

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List]:
        """Get documents by ID or filter.

        Args:
            ids: Document IDs to retrieve.
            where: Metadata filter (not yet supported, reserved).

        Returns:
            Dict with keys: ids, documents, metadatas.
        """
        result_ids = []
        result_documents = []
        result_metadatas = []

        if ids:
            for doc_id in ids:
                try:
                    resp = self._request(
                        "GET",
                        self._vectors_path(f"/{self._prefixed_id(doc_id)}"),
                        params={"namespace": self.name},
                    )
                    data = resp.json()
                    result_ids.append(doc_id)
                    result_documents.append(data.get("document", ""))
                    result_metadatas.append(data.get("vector_metadata") or {})
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise

        return {
            "ids": result_ids,
            "documents": result_documents,
            "metadatas": result_metadatas,
        }

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing documents.

        Uses upsert semantics — updates if exists, inserts if not.

        Args:
            ids: Document IDs to update.
            documents: New document texts.
            embeddings: New embedding vectors.
            metadatas: New metadata dicts.
        """
        self.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def upsert(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert documents (insert or update).

        Args:
            ids: Document IDs.
            documents: Document texts.
            embeddings: Embedding vectors.
            metadatas: Metadata dicts.
        """
        self.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.
            where: Metadata filter (reserved for future use).
        """
        if ids:
            for doc_id in ids:
                try:
                    self._request(
                        "DELETE",
                        self._vectors_path(f"/{self._prefixed_id(doc_id)}"),
                        params={"namespace": self.name},
                    )
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise

    def count(self) -> int:
        """Return the number of documents in this collection.

        Returns:
            Number of vectors in this collection's namespace.
        """
        try:
            resp = self._request(
                "GET",
                self._vectors_path("/stats"),
                params={"namespace": self.name},
            )
            data = resp.json()
            return data.get("total_count", 0)
        except requests.HTTPError:
            return 0

    def _delete_all(self) -> None:
        """Delete all vectors in this collection. Used by Client.delete_collection()."""
        try:
            resp = self._request(
                "GET",
                self._vectors_path(),
                params={"namespace": self.name, "limit": 1000},
            )
            data = resp.json()
            vectors = data.get("vectors", [])
            for vec in vectors:
                vec_id = vec.get("vector_id", "")
                if vec_id:
                    try:
                        self._request(
                            "DELETE",
                            self._vectors_path(f"/{vec_id}"),
                            params={"namespace": self.name},
                        )
                    except requests.HTTPError:
                        pass
        except requests.HTTPError:
            pass
