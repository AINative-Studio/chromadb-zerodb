"""Chroma-compatible Client backed by ZeroDB cloud."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from chromadb_zerodb.collection import Collection
from chromadb_zerodb.provision import auto_provision

DEFAULT_BASE_URL = "https://api.ainative.studio"


class Client:
    """Chroma-compatible client backed by ZeroDB cloud storage.

    Drop-in replacement for ``chromadb.Client()`` — no Docker, no server.

    Usage::

        import chromadb_zerodb as chromadb

        client = chromadb.Client()
        collection = client.create_collection("my_docs")
        collection.add(documents=["hello world"], ids=["id1"])
        results = collection.query(query_texts=["hello"], n_results=5)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self._base_url = base_url.rstrip("/")

        if api_key and project_id:
            self._api_key = api_key
            self._project_id = project_id
        else:
            self._api_key, self._project_id = auto_provision(self._base_url)

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "X-Project-ID": self._project_id,
            "Content-Type": "application/json",
            "User-Agent": "chromadb-zerodb/0.1.0",
        })

        # Track collections as namespace prefixes
        self._collections: Dict[str, Dict[str, Any]] = {}

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make an authenticated API request."""
        url = f"{self._base_url}{path}"
        resp = self._session.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Create a new collection (or return existing one).

        Args:
            name: Collection name. Used as a namespace prefix in ZeroDB.
            metadata: Optional metadata for the collection.

        Returns:
            A Collection instance.
        """
        self._collections[name] = {"metadata": metadata or {}}
        return Collection(
            name=name,
            metadata=metadata,
            session=self._session,
            base_url=self._base_url,
        )

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection by name.

        Args:
            name: Collection name.

        Returns:
            A Collection instance.
        """
        return Collection(
            name=name,
            metadata=self._collections.get(name, {}).get("metadata"),
            session=self._session,
            base_url=self._base_url,
        )

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Get or create a collection.

        Args:
            name: Collection name.
            metadata: Optional metadata (used only on creation).

        Returns:
            A Collection instance.
        """
        if name not in self._collections:
            return self.create_collection(name, metadata)
        return self.get_collection(name)

    def list_collections(self) -> List[str]:
        """List all known collection names.

        Returns:
            List of collection name strings.
        """
        return list(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Args:
            name: Collection name to delete.
        """
        collection = self.get_collection(name)
        collection._delete_all()
        self._collections.pop(name, None)

    def heartbeat(self) -> int:
        """Check connectivity to ZeroDB backend.

        Returns:
            Nanosecond timestamp if healthy.
        """
        import time
        try:
            self._request("GET", "/health")
            return int(time.time() * 1e9)
        except Exception:
            return 0
