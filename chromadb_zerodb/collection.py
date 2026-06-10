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
    ):
        self.name = name
        self.metadata = metadata or {}
        self._session = session
        self._base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._base_url}{path}"
        resp = self._session.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

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
            meta["_collection"] = self.name
            if documents[i] is not None:
                meta["_document"] = documents[i]

            self._request(
                "POST",
                "/api/v1/zerodb/vectors/upsert",
                json={
                    "id": self._prefixed_id(ids[i]),
                    "vector": embeddings[i],
                    "metadata": meta,
                    "document": documents[i] or "",
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
                "vector": embedding,
                "limit": n_results,
            }
            # Add collection filter
            filter_dict = dict(where) if where else {}
            filter_dict["_collection"] = self.name
            search_body["filter"] = filter_dict

            resp = self._request(
                "POST",
                "/api/v1/zerodb/vectors/search",
                json=search_body,
            )
            results = resp.json()

            # Normalize results format
            matches = results if isinstance(results, list) else results.get("results", results.get("matches", []))

            ids_batch = []
            docs_batch = []
            metas_batch = []
            dists_batch = []

            for match in matches:
                raw_id = match.get("id", "")
                ids_batch.append(self._strip_prefix(raw_id))
                meta = match.get("metadata", {})
                docs_batch.append(meta.pop("_document", match.get("document", "")))
                meta.pop("_collection", None)
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
                        f"/api/v1/zerodb/vectors/{self._prefixed_id(doc_id)}",
                    )
                    data = resp.json()
                    result_ids.append(doc_id)
                    meta = data.get("metadata", {})
                    result_documents.append(meta.pop("_document", data.get("document", "")))
                    meta.pop("_collection", None)
                    result_metadatas.append(meta)
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
                        f"/api/v1/zerodb/vectors/{self._prefixed_id(doc_id)}",
                    )
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise

    def count(self) -> int:
        """Return the number of documents in this collection.

        Returns:
            Number of vectors with this collection's namespace prefix.
        """
        try:
            resp = self._request("GET", "/api/v1/zerodb/vectors/stats")
            data = resp.json()
            # Stats endpoint returns total count; we approximate by
            # returning total since per-collection count requires a scan.
            return data.get("total_vectors", data.get("count", 0))
        except requests.HTTPError:
            return 0

    def _delete_all(self) -> None:
        """Delete all vectors in this collection. Used by Client.delete_collection()."""
        try:
            resp = self._request(
                "GET",
                f"/api/v1/zerodb/vectors?limit=1000",
            )
            data = resp.json()
            vectors = data if isinstance(data, list) else data.get("vectors", data.get("results", []))
            prefix = f"{self.name}::"
            for vec in vectors:
                vec_id = vec.get("id", "")
                if vec_id.startswith(prefix):
                    try:
                        self._request("DELETE", f"/api/v1/zerodb/vectors/{vec_id}")
                    except requests.HTTPError:
                        pass
        except requests.HTTPError:
            pass
