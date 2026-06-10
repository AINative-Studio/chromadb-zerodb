"""Tests for chromadb-zerodb Client and Collection."""

import json
from unittest.mock import MagicMock, patch

import pytest

from chromadb_zerodb import Client, Collection
from chromadb_zerodb.client import Client as ClientClass
from chromadb_zerodb.provision import _load_cached_credentials, _save_credentials


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "zdb_test_key_123"
FAKE_PROJECT = "proj_test_456"


@pytest.fixture
def client():
    """Create a client with fake credentials (no auto-provision)."""
    return Client(api_key=FAKE_KEY, project_id=FAKE_PROJECT, base_url="https://fake.zerodb.test")


@pytest.fixture
def collection(client):
    """Create a test collection."""
    return client.create_collection("test_docs")


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------

class TestClient:
    def test_create_collection(self, client):
        col = client.create_collection("my_col")
        assert isinstance(col, Collection)
        assert col.name == "my_col"

    def test_get_collection(self, client):
        client.create_collection("existing")
        col = client.get_collection("existing")
        assert col.name == "existing"

    def test_get_or_create_collection_new(self, client):
        col = client.get_or_create_collection("new_col")
        assert col.name == "new_col"
        assert "new_col" in client.list_collections()

    def test_get_or_create_collection_existing(self, client):
        client.create_collection("existing")
        col = client.get_or_create_collection("existing")
        assert col.name == "existing"

    def test_list_collections(self, client):
        client.create_collection("a")
        client.create_collection("b")
        names = client.list_collections()
        assert "a" in names
        assert "b" in names

    def test_delete_collection(self, client):
        client.create_collection("to_delete")
        with patch.object(Collection, "_delete_all"):
            client.delete_collection("to_delete")
        assert "to_delete" not in client.list_collections()

    def test_headers_set(self, client):
        headers = client._session.headers
        assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
        assert headers["X-Project-ID"] == FAKE_PROJECT

    def test_heartbeat_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "request", return_value=mock_resp):
            ts = client.heartbeat()
            assert ts > 0

    def test_heartbeat_failure(self, client):
        with patch.object(client._session, "request", side_effect=Exception("down")):
            ts = client.heartbeat()
            assert ts == 0


# ---------------------------------------------------------------------------
# Collection tests
# ---------------------------------------------------------------------------

class TestCollection:
    def test_prefixed_id(self, collection):
        assert collection._prefixed_id("doc1") == "test_docs::doc1"

    def test_strip_prefix(self, collection):
        assert collection._strip_prefix("test_docs::doc1") == "doc1"
        assert collection._strip_prefix("other::doc1") == "other::doc1"

    def test_add_requires_docs_or_embeddings(self, collection):
        with pytest.raises(ValueError, match="documents.*embeddings"):
            collection.add(ids=["id1"])

    def test_add_with_documents(self, collection):
        mock_embed_resp = MagicMock()
        mock_embed_resp.status_code = 200
        mock_embed_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_embed_resp.raise_for_status = MagicMock()

        mock_upsert_resp = MagicMock()
        mock_upsert_resp.status_code = 200
        mock_upsert_resp.json.return_value = {"ok": True}
        mock_upsert_resp.raise_for_status = MagicMock()

        call_count = 0

        def side_effect(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/embeddings" in url:
                return mock_embed_resp
            return mock_upsert_resp

        with patch.object(collection._session, "request", side_effect=side_effect):
            collection.add(documents=["hello world"], ids=["id1"])

        # Should have called embeddings + upsert
        assert call_count == 2

    def test_add_with_embeddings_directly(self, collection):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(collection._session, "request", return_value=mock_resp):
            collection.add(
                embeddings=[[0.1, 0.2, 0.3]],
                ids=["id1"],
                documents=["hello"],
            )

    def test_add_auto_generates_ids(self, collection):
        mock_embed_resp = MagicMock()
        mock_embed_resp.status_code = 200
        mock_embed_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_embed_resp.raise_for_status = MagicMock()

        mock_upsert_resp = MagicMock()
        mock_upsert_resp.status_code = 200
        mock_upsert_resp.json.return_value = {"ok": True}
        mock_upsert_resp.raise_for_status = MagicMock()

        def side_effect(method, url, **kwargs):
            if "/embeddings" in url:
                return mock_embed_resp
            return mock_upsert_resp

        with patch.object(collection._session, "request", side_effect=side_effect):
            # Should not raise — IDs auto-generated
            collection.add(documents=["doc1"])

    def test_query_requires_input(self, collection):
        with pytest.raises(ValueError, match="query_texts or query_embeddings"):
            collection.query()

    def test_query_with_texts(self, collection):
        mock_embed_resp = MagicMock()
        mock_embed_resp.status_code = 200
        mock_embed_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_embed_resp.raise_for_status = MagicMock()

        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {
            "results": [
                {
                    "id": "test_docs::id1",
                    "metadata": {"_document": "hello world", "_collection": "test_docs", "topic": "greetings"},
                    "distance": 0.05,
                },
                {
                    "id": "test_docs::id2",
                    "metadata": {"_document": "goodbye world", "_collection": "test_docs"},
                    "distance": 0.15,
                },
            ]
        }
        mock_search_resp.raise_for_status = MagicMock()

        def side_effect(method, url, **kwargs):
            if "/embeddings" in url:
                return mock_embed_resp
            return mock_search_resp

        with patch.object(collection._session, "request", side_effect=side_effect):
            results = collection.query(query_texts=["hello"], n_results=2)

        assert results["ids"] == [["id1", "id2"]]
        assert results["documents"] == [["hello world", "goodbye world"]]
        assert results["distances"] == [[0.05, 0.15]]
        # Internal metadata keys should be stripped
        assert "_collection" not in results["metadatas"][0][0]
        assert "_document" not in results["metadatas"][0][0]
        assert results["metadatas"][0][0].get("topic") == "greetings"

    def test_query_with_embeddings(self, collection):
        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {"results": []}
        mock_search_resp.raise_for_status = MagicMock()

        with patch.object(collection._session, "request", return_value=mock_search_resp):
            results = collection.query(query_embeddings=[[0.1, 0.2]], n_results=5)

        assert results["ids"] == [[]]

    def test_get_by_ids(self, collection):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test_docs::id1",
            "metadata": {"_document": "hello", "_collection": "test_docs", "key": "val"},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(collection._session, "request", return_value=mock_resp):
            result = collection.get(ids=["id1"])

        assert result["ids"] == ["id1"]
        assert result["documents"] == ["hello"]
        assert result["metadatas"][0]["key"] == "val"

    def test_get_missing_id_skipped(self, collection):
        error_resp = MagicMock()
        error_resp.status_code = 404
        http_error = MagicMock(spec=Exception)
        exc = Exception("404")
        exc.response = error_resp
        from requests.exceptions import HTTPError
        exc = HTTPError(response=error_resp)

        with patch.object(collection._session, "request", side_effect=exc):
            result = collection.get(ids=["nonexistent"])

        assert result["ids"] == []

    def test_delete_by_ids(self, collection):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(collection._session, "request", return_value=mock_resp) as mock_req:
            collection.delete(ids=["id1", "id2"])

        assert mock_req.call_count == 2

    def test_delete_missing_id_ignored(self, collection):
        from requests.exceptions import HTTPError
        error_resp = MagicMock()
        error_resp.status_code = 404
        exc = HTTPError(response=error_resp)

        with patch.object(collection._session, "request", side_effect=exc):
            # Should not raise
            collection.delete(ids=["nonexistent"])

    def test_count(self, collection):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total_vectors": 42}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(collection._session, "request", return_value=mock_resp):
            assert collection.count() == 42

    def test_count_on_error(self, collection):
        from requests.exceptions import HTTPError
        error_resp = MagicMock()
        error_resp.status_code = 500
        exc = HTTPError(response=error_resp)

        with patch.object(collection._session, "request", side_effect=exc):
            assert collection.count() == 0

    def test_upsert_delegates_to_add(self, collection):
        with patch.object(collection, "add") as mock_add:
            collection.upsert(ids=["id1"], documents=["doc1"])
            mock_add.assert_called_once_with(
                ids=["id1"], documents=["doc1"], embeddings=None, metadatas=None
            )

    def test_update_delegates_to_add(self, collection):
        with patch.object(collection, "add") as mock_add:
            collection.update(ids=["id1"], documents=["doc1"])
            mock_add.assert_called_once_with(
                ids=["id1"], documents=["doc1"], embeddings=None, metadatas=None
            )


# ---------------------------------------------------------------------------
# Provision tests
# ---------------------------------------------------------------------------

class TestProvision:
    def test_load_cached_credentials_missing(self, tmp_path):
        with patch("chromadb_zerodb.provision.CREDENTIALS_PATH", tmp_path / "creds.json"):
            key, pid = _load_cached_credentials()
            assert key is None
            assert pid is None

    def test_save_and_load_credentials(self, tmp_path):
        creds_path = tmp_path / "creds.json"
        with patch("chromadb_zerodb.provision.CREDENTIALS_PATH", creds_path):
            _save_credentials("key123", "proj456")
            key, pid = _load_cached_credentials()
            assert key == "key123"
            assert pid == "proj456"

    def test_auto_provision_from_env(self):
        with patch.dict("os.environ", {"ZERODB_API_KEY": "envkey", "ZERODB_PROJECT_ID": "envproj"}):
            from chromadb_zerodb.provision import auto_provision
            key, pid = auto_provision()
            assert key == "envkey"
            assert pid == "envproj"


# ---------------------------------------------------------------------------
# Integration-style test (full flow with mocks)
# ---------------------------------------------------------------------------

class TestIntegrationFlow:
    def test_full_add_query_flow(self):
        """Simulate: create client -> create collection -> add -> query."""
        client = Client(api_key=FAKE_KEY, project_id=FAKE_PROJECT, base_url="https://fake.test")
        col = client.create_collection("knowledge")

        mock_upsert_resp = MagicMock()
        mock_upsert_resp.status_code = 200
        mock_upsert_resp.json.return_value = {"ok": True}
        mock_upsert_resp.raise_for_status = MagicMock()

        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {
            "results": [
                {
                    "id": "knowledge::d1",
                    "metadata": {"_document": "Python is great", "_collection": "knowledge"},
                    "distance": 0.02,
                },
            ]
        }
        mock_search_resp.raise_for_status = MagicMock()

        call_log = []

        def side_effect(method, url, **kwargs):
            call_log.append((method, url))
            if "/embeddings" in url:
                # Return correct number of embeddings based on input
                body = kwargs.get("json", {})
                inputs = body.get("input", [])
                n = len(inputs) if isinstance(inputs, list) else 1
                embed_resp = MagicMock()
                embed_resp.status_code = 200
                embed_resp.json.return_value = {
                    "data": [{"embedding": [0.1 * (i + 1), 0.2, 0.3]} for i in range(n)]
                }
                embed_resp.raise_for_status = MagicMock()
                return embed_resp
            if "/search" in url:
                return mock_search_resp
            return mock_upsert_resp

        with patch.object(col._session, "request", side_effect=side_effect):
            col.add(
                documents=["Python is great", "JavaScript too"],
                ids=["d1", "d2"],
            )
            results = col.query(query_texts=["best language"], n_results=1)

        assert results["ids"] == [["d1"]]
        assert results["documents"] == [["Python is great"]]
        assert len(call_log) >= 4  # embed + 2 upserts + embed + search
