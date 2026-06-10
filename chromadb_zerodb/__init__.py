"""chromadb-zerodb: Chroma-compatible API backed by ZeroDB cloud."""

from chromadb_zerodb.client import Client
from chromadb_zerodb.collection import Collection

__version__ = "0.1.0"
__all__ = ["Client", "Collection"]
