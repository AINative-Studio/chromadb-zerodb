# chromadb-zerodb

**Chroma-compatible API backed by ZeroDB cloud. No Docker, no server, instant vector DB.**

Drop-in replacement for `chromadb` — same `Collection` API, but vectors are stored in ZeroDB cloud. Auto-provisions on first use. Free tier included.

## Why?

| | chromadb | chromadb-zerodb |
|---|---------|----------------|
| **Setup** | Docker server or in-memory (lost on restart) | `pip install` and go |
| **Persistence** | Requires server config | Cloud-persisted automatically |
| **Embeddings** | Bring your own | Free BAAI BGE models included |
| **Scaling** | Self-managed | Managed infrastructure |
| **Cost** | Server hosting | Free tier, pay as you grow |

## Install

```bash
pip install chromadb-zerodb
```

## Quick Start

```python
import chromadb_zerodb as chromadb

# Auto-provisions a free project on first use
client = chromadb.Client()

# Create a collection
collection = client.create_collection("my_docs")

# Add documents (auto-embedded with free BGE models)
collection.add(
    documents=[
        "Python is a great programming language",
        "JavaScript powers the web",
        "Rust is fast and memory-safe",
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"topic": "python"},
        {"topic": "javascript"},
        {"topic": "rust"},
    ],
)

# Query by text
results = collection.query(
    query_texts=["best language for beginners"],
    n_results=2,
)

print(results["documents"])  # [["Python is a great...", "JavaScript powers..."]]
print(results["distances"])  # [[0.05, 0.12]]
```

## Migration from chromadb

Change one import:

```python
# Before
import chromadb
client = chromadb.Client()

# After
import chromadb_zerodb as chromadb
client = chromadb.Client()
```

Everything else stays the same.

## Authentication

Three ways to authenticate (checked in order):

1. **Constructor arguments:**
   ```python
   client = chromadb.Client(api_key="zdb_...", project_id="proj_...")
   ```

2. **Environment variables:**
   ```bash
   export ZERODB_API_KEY=zdb_...
   export ZERODB_PROJECT_ID=proj_...
   ```

3. **Auto-provision:** If neither is set, a free project is created automatically. Credentials are cached in `~/.zerodb/credentials.json`.

## API Reference

### Client

```python
client = chromadb.Client(api_key=None, project_id=None, base_url=None)

client.create_collection(name, metadata=None)    # -> Collection
client.get_collection(name)                       # -> Collection
client.get_or_create_collection(name, metadata=None)  # -> Collection
client.list_collections()                         # -> List[str]
client.delete_collection(name)                    # -> None
client.heartbeat()                                # -> int (nanosecond timestamp)
```

### Collection

```python
collection.add(
    ids=None,           # auto-generated if omitted
    documents=None,     # auto-embedded if provided
    embeddings=None,    # or pass pre-computed vectors
    metadatas=None,
)

collection.query(
    query_texts=None,
    query_embeddings=None,
    n_results=10,
    where=None,         # metadata filter
)
# Returns: {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}

collection.get(ids=None, where=None)
# Returns: {"ids": [...], "documents": [...], "metadatas": [...]}

collection.update(ids, documents=None, embeddings=None, metadatas=None)
collection.upsert(ids, documents=None, embeddings=None, metadatas=None)
collection.delete(ids=None, where=None)
collection.count()  # -> int
```

## Bring Your Own Embeddings

```python
collection.add(
    ids=["id1"],
    embeddings=[[0.1, 0.2, 0.3, ...]],  # your vectors
    documents=["the original text"],
    metadatas=[{"source": "my_model"}],
)

collection.query(
    query_embeddings=[[0.1, 0.2, 0.3, ...]],
    n_results=5,
)
```

## Metadata Filtering

```python
results = collection.query(
    query_texts=["search term"],
    where={"topic": "python"},
    n_results=5,
)
```

## How It Works

- **Collections** are namespaced via ID prefixes (`collection_name::doc_id`)
- **Auto-embedding** uses ZeroDB's free BAAI BGE-M3 model
- **Storage** is ZeroDB's managed vector database (same infra as ZeroDB MCP)
- **Credentials** are cached at `~/.zerodb/credentials.json`

## Links

- [ZeroDB Documentation](https://docs.ainative.studio)
- [ZeroDB MCP Server](https://pypi.org/project/zerodb-mcp/)
- [AINative Studio](https://ainative.studio)

## License

MIT

---

## Powered by ZeroDB + AINative

This package is part of the [AINative](https://ainative.studio) ecosystem — the AI-native developer platform.

### Why ZeroDB?

| Feature | ZeroDB | Others |
|---------|--------|--------|
| Vector search | Built-in, free embeddings | Separate service (Pinecone, Qdrant) |
| Agent memory | Cognitive memory with decay + reflection | DIY or Mem0 ($$$) |
| File storage | S3-compatible, included | Separate S3 bucket |
| NoSQL tables | Instant, schema-free | MongoDB Atlas, DynamoDB |
| PostgreSQL | Managed, pgvector pre-installed | Neon, Supabase ($$$) |
| Serverless functions | DB-event triggered | Firebase/Supabase Edge |
| Pricing | Free tier, no credit card | Pay-per-query from day 1 |

### Get Started Free

```bash
npx zerodb-cli init    # Auto-configures your IDE
```

Or sign up at **[ainative.studio](https://ainative.studio)** — free tier, no credit card required.

[View all ZeroDB packages →](https://docs.ainative.studio)

