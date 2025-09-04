# app/opensearch_service.py
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import AlibabaCloudOpenSearch, AlibabaCloudOpenSearchSettings
from app.gemini_service import embed_texts


class GeminiEmbeddingAdapter:
    """Thin adapter so LangChain vector store can call our Gemini embeddings."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return embed_texts([text])[0]


# ----- Read settings from environment -----
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INSTANCE_ID = os.getenv("OPENSEARCH_INSTANCE_ID")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
OPENSEARCH_NAMESPACE = os.getenv("OPENSEARCH_NAMESPACE", "")
OPENSEARCH_TABLE = os.getenv("OPENSEARCH_TABLE")
FIELD_NAME_ID = os.getenv("OPENSEARCH_FIELD_ID", "id")
FIELD_NAME_TEXT = os.getenv("OPENSEARCH_FIELD_TEXT", "document")
FIELD_NAME_EMB = os.getenv("OPENSEARCH_FIELD_EMBEDDING", "embedding")
FIELD_MAP_EXTRA = {}  # can include mappings for title, source, etc.


def _settings() -> AlibabaCloudOpenSearchSettings:
    return AlibabaCloudOpenSearchSettings(
        endpoint=OPENSEARCH_ENDPOINT,
        instance_id=OPENSEARCH_INSTANCE_ID,
        protocol="http",
        username=OPENSEARCH_USERNAME,
        password=OPENSEARCH_PASSWORD,
        namespace=OPENSEARCH_NAMESPACE if OPENSEARCH_NAMESPACE else None,
        table_name=OPENSEARCH_TABLE,
        embedding_field_separator=",",
        output_fields=None,
        field_name_mapping={
            "id": FIELD_NAME_ID,
            "document": FIELD_NAME_TEXT,
            "embedding": FIELD_NAME_EMB,
            # make sure 'title' is mapped if you use it
            "title": "title",
            **FIELD_MAP_EXTRA,
        },
    )


def _get_store() -> AlibabaCloudOpenSearch:
    emb = GeminiEmbeddingAdapter()
    return AlibabaCloudOpenSearch(embedding=emb, config=_settings())


def upsert_vectors(items: list[dict]):
    store = _get_store()

    texts, ids, metadatas = [], [], []

    for item in items:
        text = item.get("text") or ""
        if not text:
            continue
        texts.append(text)
        ids.append(item.get("id"))

        meta = item.get("metadata", {}) or {}

        # Ensure all required keys exist for OpenSearch
        meta.setdefault("url", "")      # Prevent KeyError
        meta.setdefault("title", f"doc-{item.get('id')}")
        meta.setdefault("source", "")

        metadatas.append(meta)

    store.add_texts(texts=texts, ids=ids, metadatas=metadatas)



def query_vector(vec: List[float], top_k: int = 4):
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
        _id = d.metadata.get("id") or d.metadata.get(FIELD_NAME_ID) or ""
        _score = d.metadata.get("_score") or d.metadata.get("score") or 0.0
        meta = dict(d.metadata or {})
        meta["text"] = d.page_content
        # Ensure title exists
        if "title" not in meta:
            meta["title"] = f"doc-{_id}"
        matches.append(Match(_id, _score, meta))

    return Result(matches)
