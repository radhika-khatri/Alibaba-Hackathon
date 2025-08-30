# app/opensearch_service.py
import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import (
    AlibabaCloudOpenSearch,
    AlibabaCloudOpenSearchSettings,
)

# We’ll reuse your Qwen embedding endpoint to keep everything Alibaba-native
from app.qwen_service import qwen_embed


class QwenEmbeddingAdapter:
    """Thin adapter so LangChain vector store can call our Qwen embeddings."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return qwen_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return qwen_embed([text])[0]


# ----- Read settings from environment -----
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")        # e.g. ha-cn-xxxxxx.public.ha.aliyuncs.com
OPENSEARCH_INSTANCE_ID = os.getenv("OPENSEARCH_INSTANCE_ID")  # e.g. ha-cn-xxxxxx
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

# Data partition & table config created in your OpenSearch Vector Search Edition instance
OPENSEARCH_NAMESPACE = os.getenv("OPENSEARCH_NAMESPACE", "")  # optional; leave blank if not enabled
OPENSEARCH_TABLE = os.getenv("OPENSEARCH_TABLE")              # e.g. kb_table

# Field mapping must match the schema you defined in the OpenSearch console
# Fields: id (pk), document (text), embedding (vector), plus optional metadata
FIELD_NAME_ID = os.getenv("OPENSEARCH_FIELD_ID", "id")
FIELD_NAME_TEXT = os.getenv("OPENSEARCH_FIELD_TEXT", "document")
FIELD_NAME_EMB = os.getenv("OPENSEARCH_FIELD_EMBEDDING", "embedding")

# Example extra metadata fields; ensure they exist in your table schema
# You can add/remove mappings according to your OpenSearch table
FIELD_MAP_EXTRA = {
    # "title": "title,=",
    # "url": "url,=",
    # "score": "score,=",
}


def _settings() -> AlibabaCloudOpenSearchSettings:
    return AlibabaCloudOpenSearchSettings(
        endpoint=OPENSEARCH_ENDPOINT,
        instance_id=OPENSEARCH_INSTANCE_ID,
        protocol="http",  # or "https" if enabled
        username=OPENSEARCH_USERNAME,
        password=OPENSEARCH_PASSWORD,
        namespace=OPENSEARCH_NAMESPACE if OPENSEARCH_NAMESPACE else None,
        tablename=OPENSEARCH_TABLE,
        embedding_field_separator=",",  # default
        output_fields=None,  # default: mapped fields returned
        field_name_mapping={
            "id": FIELD_NAME_ID,
            "document": FIELD_NAME_TEXT,
            "embedding": FIELD_NAME_EMB,
            **FIELD_MAP_EXTRA,
        },
    )


def _get_store() -> AlibabaCloudOpenSearch:
    emb = QwenEmbeddingAdapter()
    return AlibabaCloudOpenSearch(embedding=emb, config=_settings())


def upsert_vectors(items: List[Dict[str, Any]]) -> None:
    """
    items: list of dicts like:
      {
        "id": "doc123",
        "values": [float, ...],  # optional; if omitted we’ll re-embed from text
        "metadata": {"title": "...", "url": "...", "text": "..."}
      }
    We store page content as the 'text' field and keep metadata.
    """
    store = _get_store()

    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for it in items:
        meta = it.get("metadata", {}) or {}
        text = meta.get("text") or ""
        if not text:
            continue
        texts.append(text)
        ids.append(it.get("id"))
        # You can include any extra fields that exist in your OpenSearch table schema
        m = {k: v for k, v in meta.items() if k != "text"}
        metadatas.append(m)

    # This will call Qwen embeddings internally via QwenEmbeddingAdapter
    # and create/update documents in the OpenSearch table.
    store.add_texts(texts=texts, ids=ids, metadatas=metadatas)


def query_vector(vec: List[float], top_k: int = 4):
    """
    Returns an object with 'matches' to be drop-in compatible with your previous code.
    We translate LangChain Documents to a Pinecone-like shape.
    """
    store = _get_store()
    docs = store.similarity_search_by_vector(vec, k=top_k)

    class Match:
        def __init__(self, _id, _score, _meta):
            self.id = _id
            self.score = _score
            self.metadata = _meta

    class Result:
        def __init__(self, matches):
            self.matches = matches

    matches = []
    for d in docs:
        # LC Document: d.page_content, d.metadata, d.id (id may be in metadata depending on mapping)
        _id = d.metadata.get("id") or d.metadata.get(FIELD_NAME_ID) or ""
        _score = d.metadata.get("_score") or d.metadata.get("score") or 0.0
        meta = dict(d.metadata or {})
        # ensure original text is available to build snippet
        meta["text"] = d.page_content
        matches.append(Match(_id, _score, meta))

    return Result(matches)
